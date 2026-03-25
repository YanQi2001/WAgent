"""Interactive interview session – ties Harness + LangGraph + Agents together for CLI use."""

from __future__ import annotations

import logging

import pdfplumber
from rich.console import Console
from rich.panel import Panel

from wagent.cli.prompt_utils import prompt_input

from wagent.agents.graph import build_interview_graph, create_graph_agent
from wagent.agents.interviewer import evaluate_answer, generate_question
from wagent.agents.schemas import InterviewPlan
from wagent.harness.harness import InterviewHarness
from wagent.harness.state import InterviewState, QAPair, QuestionMode
from wagent.rag.retriever import HybridRetriever
from wagent.rag.store import get_qdrant_client, ensure_collection, collection_stats, COLLECTION_NAME

logger = logging.getLogger(__name__)
console = Console()


def parse_resume_pdf(path: str) -> str:
    """Extract text from a resume PDF using pdfplumber."""
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
    return "\n".join(text_parts)


SYSTEM_PROMPT = """\
你是一位资深的 AI/大模型技术面试官。你的任务是通过结构化的提问评估候选人在
AI、大语言模型和 Agent 系统方面的技术能力。

面试规则：
1. 每次只问一个明确、聚焦的问题
2. 使用中文提问
3. 根据候选人回答质量决定追问深度：好的回答深挖1-2层，差的回答跳到下一个话题
4. 保持专业和鼓励的态度
5. 面试过程中追踪已覆盖的知识点
"""


MAX_FOLLOWUP_DEPTH = 3
CONSECUTIVE_LOW_THRESHOLD = 2
LOW_SCORE_CUTOFF = 4.0
DEEP_FOLLOWUP_SCORE = 7.0


def _retrieve_context(retriever: HybridRetriever, topic: str) -> str:
    """Retrieve knowledge context for a topic via hybrid search."""
    try:
        results = retriever.retrieve(topic, topic_filter=None)
        if results:
            snippets = [r["text"][:300] for r in results[:3]]
            return "\n---\n".join(snippets)
    except Exception as e:
        logger.debug("RAG retrieval failed for topic '%s': %s", topic, e)
    return ""


async def run_interview(resume_path: str) -> None:
    """Run a full interactive interview session."""
    console.print(Panel("[bold]AI 面试官系统[/bold] — Harness-Centric Architecture", style="blue"))

    # --- 1. Parse resume ---
    console.print(f"\n[dim]解析简历: {resume_path}[/dim]")
    resume_text = parse_resume_pdf(resume_path)
    if not resume_text.strip():
        console.print("[red]简历解析失败或内容为空[/red]")
        return
    console.print(f"[green]简历解析完成[/green] ({len(resume_text)} chars)")

    # --- 2. Initialize state ---
    state = InterviewState()

    # --- 3. Build LangGraph compiled graph and route resume ---
    console.print("\n[dim]构建 LangGraph 面试图...[/dim]")
    compiled_graph = build_interview_graph()

    console.print("[dim]分析简历，生成面试计划（via LangGraph route_node）...[/dim]")
    route_result = await compiled_graph.ainvoke({
        "interview_state": state,
        "resume_text": resume_text,
        "phase": "routing",
    })
    plan: InterviewPlan = route_result["plan"]
    console.print(
        f"[green]面试计划生成完成[/green]\n"
        f"  简历相关话题: {plan.resume_topics}\n"
        f"  随机八股话题: {plan.random_topics}\n"
        f"  计划题数: {plan.resume_question_count}(简历) + {plan.random_question_count}(随机)"
    )

    # --- 3.3. Resume GAP analysis ---
    try:
        from wagent.mcp_servers.updater import KnowledgeUpdater
        updater = KnowledgeUpdater()
        console.print("\n[dim]分析简历知识点覆盖情况...[/dim]")
        weak_topics = await updater.resume_gap_analysis(plan.resume_topics)
        if weak_topics:
            console.print("[yellow]发现以下知识点在知识库中覆盖不足：[/yellow]")
            for wt in weak_topics:
                topic = wt.get("topic", "")
                count = wt.get("current_count", 0)
                keywords = wt.get("search_keywords", [])
                console.print(f"  • {topic} ({count} chunks) → 搜索词: {keywords}")

            from wagent.cli.smart_prompt import prompt_and_parse

            topic_names = [wt.get("topic", "") for wt in weak_topics]
            decision = await prompt_and_parse(
                "[bold green]是否搜索补全？[/bold green]（可输入：全部搜索 / 只搜索XX和YY / 跳过）",
                situation=f"知识库中以下话题覆盖不足: {topic_names}。用户需要决定是否搜索补全。",
                actions=[
                    {"id": "fill_all", "desc": "搜索全部缺口话题"},
                    {"id": "fill_selected", "desc": "只搜索用户指定的话题", "has_items": True},
                    {"id": "skip", "desc": "跳过，不搜索"},
                ],
                available_items=topic_names,
            )

            if decision["action"] == "skip":
                console.print("[dim]已记录待补全话题，将在下次定时更新时优先处理[/dim]")
            else:
                selected_topics = decision["params"].get("items", []) if decision["action"] == "fill_selected" else None

                all_keywords = []
                for wt in weak_topics:
                    if selected_topics is None or wt.get("topic", "") in selected_topics:
                        all_keywords.extend(wt.get("search_keywords", []))
                if all_keywords:
                    if selected_topics:
                        console.print(f"[dim]仅搜索: {selected_topics}[/dim]")
                    console.print("[dim]正在搜索并补全知识库...[/dim]")
                    fill_report = await updater.fill_gaps(all_keywords)
                    console.print(
                        f"[green]补全完成[/green]: 搜索 {fill_report['searched']} 条, "
                        f"提取 {fill_report['extracted']} 条, 入库 {fill_report['added']} 条"
                    )
                else:
                    console.print("[dim]没有可用的搜索关键词[/dim]")
        else:
            console.print("[green]知识库对简历相关话题覆盖良好[/green]")
    except Exception as e:
        logger.warning("GAP analysis skipped: %s", e)

    # --- 3.5. Initialize RAG retriever ---
    retriever = HybridRetriever()
    try:
        client = get_qdrant_client()
        ensure_collection(client)
        total = collection_stats(client).get("total_points") or 2000
        points, _ = client.scroll(
            collection_name=COLLECTION_NAME, limit=total, with_payload=True
        )
        if points:
            docs = [{"text": p.payload.get("text", ""), **p.payload} for p in points if p.payload.get("text")]
            retriever.build_bm25_index(docs)
            console.print(f"[green]知识库已加载[/green] ({len(docs)} chunks, BM25 索引已构建)")
        else:
            console.print("[yellow]知识库为空，面试官将不使用知识库内容出题[/yellow]")
    except Exception as e:
        console.print(f"[yellow]知识库加载失败: {e}[/yellow]")

    # --- 3.6. Set up harness with graph agent ---
    harness = InterviewHarness()
    graph_agent = create_graph_agent(compiled_graph)
    harness.set_agent(graph_agent)
    harness.set_system_prompt(SYSTEM_PROMPT)

    current_topic = plan.resume_topics[0] if plan.resume_topics else "general"
    harness.set_agent_context(plan=plan, retriever=retriever, current_topic=current_topic)

    # --- 4. Generate first question ---
    console.print("\n" + "=" * 60)
    console.print("[bold]面试开始！[/bold] 输入 /end 随时结束面试\n")

    knowledge_context = _retrieve_context(retriever, current_topic)
    question = await generate_question(state, plan, knowledge_context=knowledge_context)
    console.print(Panel(question, title="面试官", style="cyan"))

    # --- 5. Interview loop (Harness outer loop) ---
    current_followup_depth = 0
    consecutive_low_count = 0

    while True:
        answer = await prompt_input("你的回答", "interview")

        if answer.strip().lower() == "/end":
            state.progress.current_phase = "ending"
            console.print("\n[yellow]候选人主动结束面试[/yellow]")
            break

        response = await harness.turn(state, answer)

        evaluation = await evaluate_answer(question, answer, current_topic)

        if evaluation.is_exit_request:
            state.progress.current_phase = "ending"
            console.print("\n[yellow]检测到退出意图，面试结束[/yellow]")
            break

        qa = QAPair(
            question=question,
            answer=answer,
            topic=current_topic,
            mode=state.progress.current_mode,
            follow_up_depth=current_followup_depth,
        )
        qa.score = evaluation.score
        state.qa_history.append(qa)
        state.progress.questions_asked += 1

        if current_topic not in state.progress.covered_topics:
            state.progress.covered_topics.append(current_topic)

        should_end = await harness.suggest_end(state)
        if should_end:
            console.print("\n[yellow]系统建议结束面试（考点覆盖率已达标）[/yellow]")
            state.progress.current_phase = "ending"
            break

        # --- Adaptive follow-up decision ---
        switch_topic = False
        switch_reason = ""

        if evaluation.score < LOW_SCORE_CUTOFF:
            consecutive_low_count += 1
            if consecutive_low_count >= CONSECUTIVE_LOW_THRESHOLD:
                switch_topic = True
                switch_reason = f"连续 {consecutive_low_count} 轮低分"
                consecutive_low_count = 0
            else:
                switch_topic = True
                switch_reason = "得分较低"
        elif evaluation.score >= DEEP_FOLLOWUP_SCORE and evaluation.should_follow_up:
            if current_followup_depth < MAX_FOLLOWUP_DEPTH:
                current_followup_depth += 1
                consecutive_low_count = 0
            else:
                switch_topic = True
                switch_reason = f"已追问 {MAX_FOLLOWUP_DEPTH} 轮"
                consecutive_low_count = 0
        elif evaluation.score >= LOW_SCORE_CUTOFF and evaluation.should_follow_up:
            consecutive_low_count = 0
        else:
            switch_topic = True
            switch_reason = "该方向已充分考察"
            consecutive_low_count = 0

        if switch_topic:
            current_followup_depth = 0
            if switch_reason:
                console.print(f"[dim]→ 切换方向（{switch_reason}）[/dim]")

            if state.progress.current_mode == QuestionMode.RESUME_DRIVEN:
                pending = [t for t in plan.resume_topics if t not in state.progress.covered_topics]
                current_topic = pending[0] if pending else plan.random_topics[0] if plan.random_topics else "general"
            else:
                remaining_random = [t for t in plan.random_topics if t not in state.progress.covered_topics]
                current_topic = remaining_random[0] if remaining_random else "general"

        harness.set_agent_context(current_topic=current_topic)

        knowledge_context = _retrieve_context(retriever, current_topic)
        question = response if response and not response.startswith("[面试系统]") else await generate_question(state, plan, knowledge_context=knowledge_context)
        console.print(Panel(question, title="面试官", style="cyan"))

        console.print(
            f"[dim]进度: {state.progress.questions_asked}题 | "
            f"方向: {current_topic} | "
            f"追问深度: {current_followup_depth}/{MAX_FOLLOWUP_DEPTH} | "
            f"覆盖: {state.covered_topic_ratio:.0%} | "
            f"{harness.budget.budget_status(state)}[/dim]"
        )

    # --- 6. Judge scoring (via LangGraph judge_node) ---
    console.print("\n" + "=" * 60)
    console.print("[dim]评分中（Judge Agent via LangGraph judge_node）...[/dim]")
    judge_result = await compiled_graph.ainvoke({
        "interview_state": state,
        "phase": "judging",
    })
    scorecard = judge_result["scorecard"]

    scorecard_text = (
        f"总分: [bold]{scorecard.overall_score}/10[/bold]\n"
        f"推荐: {scorecard.recommendation}\n\n"
        f"优势:\n" + "\n".join(f"  + {s}" for s in scorecard.strengths) + "\n\n"
        f"不足:\n" + "\n".join(f"  - {w}" for w in scorecard.weaknesses) + "\n\n"
        f"--- 深度评估维度 ---\n"
        f"实战经验指数: {scorecard.battle_scars_index}/10\n"
        f"第一性原理思维: {scorecard.first_principles_score}/10\n"
        f"STAR 完整度: {scorecard.star_completeness}/10\n"
        f"追问韧性: {scorecard.followup_resilience}/10\n\n"
        f"总结: {scorecard.summary}"
    )
    if scorecard.deep_analysis:
        scorecard_text += f"\n\n深度分析:\n{scorecard.deep_analysis}"
    console.print(Panel(scorecard_text, title="评分报告", style="green"))

    # --- 6.5. Study guide ---
    if scorecard.study_guide:
        from rich.table import Table

        study_table = Table(title="薄弱知识点学习建议", show_lines=True)
        study_table.add_column("优先级", style="bold", justify="center", width=8)
        study_table.add_column("方向", style="cyan", width=16)
        study_table.add_column("薄弱点", width=36)
        study_table.add_column("建议搜索关键词", style="green", width=30)

        priority_style = {"high": "[bold red]高[/bold red]", "medium": "[yellow]中[/yellow]", "low": "[dim]低[/dim]"}
        for item in scorecard.study_guide:
            study_table.add_row(
                priority_style.get(item.priority, item.priority),
                item.topic,
                item.weakness_summary,
                "、".join(item.suggested_keywords) if item.suggested_keywords else "—",
            )

        console.print()
        console.print(study_table)
        console.print(
            "\n[dim]💡 提示: 可以将以上关键词直接输入 [cyan]wagent qa[/cyan] 模式进行学习，"
            "或使用 [cyan]wagent prepare[/cyan] 自动搜索补全知识库。[/dim]"
        )

    # --- 7. Finalize ---
    summary = harness.finalize(state)
    console.print(f"\n[dim]Trace 已保存: {summary.get('trace_file', 'N/A')}[/dim]")
    console.print(f"[dim]{summary.get('budget', '')}[/dim]")
