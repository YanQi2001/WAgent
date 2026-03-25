"""WAgent CLI – AI Interview Agent System."""

from __future__ import annotations

import atexit
import asyncio
import logging

import typer
from rich.console import Console
from rich.prompt import Confirm

from wagent.cli.prompt_utils import prompt_input

app = typer.Typer(name="wagent", help="AI Interview Agent with Harness-Centric Architecture")
console = Console()


def _setup_logging(verbose: bool) -> None:
    """Configure logging with optional third-party noise suppression."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        for noisy in (
            "httpx", "httpcore", "sentence_transformers",
            "huggingface_hub", "transformers", "urllib3", "filelock",
        ):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def _cleanup():
    try:
        from wagent.rag.store import close_qdrant_client
        close_qdrant_client()
    except Exception:
        pass


atexit.register(_cleanup)


@app.command()
def ping():
    """Verify that the LLM API connection works."""
    from wagent.llm import get_llm

    console.print("[bold]Testing LLM API connection...[/bold]")
    llm = get_llm(max_tokens=64)
    resp = llm.invoke("Say 'connection ok' in exactly two words.")
    console.print(f"[green]Response:[/green] {resp.content}")


@app.command()
def start(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
):
    """Unified entry point with LLM-based intent recognition.

    Analyzes the user's first message to determine mode (interview / qa / chitchat),
    asks for confirmation, then enters the appropriate session.
    """
    _setup_logging(verbose)

    async def _run():
        from wagent.agents.intent import classify_intent, INTENT_LABELS

        console.print("[bold]欢迎使用 AI 面试官系统！[/bold]")
        console.print("请输入任意内容，系统将自动识别您的意图。\n")
        console.print("[dim]提示: 输入面试相关内容进入面试模式，输入技术问题进入问答模式[/dim]\n")

        user_input = await prompt_input("请输入")

        if not user_input.strip():
            console.print("[yellow]未检测到输入，退出。[/yellow]")
            return

        console.print("[dim]正在分析意图...[/dim]")
        result = classify_intent(user_input)
        if asyncio.iscoroutine(result):
            result = await result

        intent = result["intent"]
        label = INTENT_LABELS.get(intent, intent)
        confidence = result.get("confidence", 0)

        console.print(
            f"\n识别结果: [bold]{label}[/bold] (置信度: {confidence:.0%})"
        )

        from wagent.cli.smart_prompt import prompt_and_parse

        if intent == "chitchat":
            situation = "用户输入看起来是闲聊，系统需要确认用户想进入哪个模式。"
        else:
            situation = f"系统识别用户意图为「{label}」（置信度 {confidence:.0%}），需要用户确认或切换模式。"

        decision = await prompt_and_parse(
            f"[bold green]确认进入{label}？[/bold green]（可输入：确认 / 面试 / 问答 / 退出）",
            situation=situation,
            actions=[
                {"id": "interview", "desc": "进入面试模式（需要提供简历）"},
                {"id": "qa", "desc": "进入知识问答模式"},
                {"id": "exit", "desc": "退出系统"},
            ],
        )
        intent = decision["action"] if decision["action"] != "exit" else None

        if intent == "interview":
            resume_path = await prompt_input("请输入简历 PDF 路径")
            from wagent.cli.interview_session import run_interview
            await run_interview(resume_path)
        elif intent == "qa":
            from wagent.cli.qa_session import run_qa_session
            await run_qa_session()
        else:
            console.print("[yellow]已退出[/yellow]")

    asyncio.run(_run())


@app.command()
def interview(
    resume: str = typer.Argument(..., help="Path to candidate resume PDF"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
):
    """Start an interactive interview session (direct entry, no intent recognition)."""
    _setup_logging(verbose)

    from wagent.cli.interview_session import run_interview

    asyncio.run(run_interview(resume))


@app.command()
def qa(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
):
    """Start an interactive Q&A session (direct entry, no intent recognition)."""
    _setup_logging(verbose)

    from wagent.cli.qa_session import run_qa_session

    asyncio.run(run_qa_session())


@app.command()
def ingest(
    source: str = typer.Option("manual", help="Source type: manual or crawled"),
    path: str = typer.Option("data/documents", help="Directory containing PDFs"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
    daemon: bool = typer.Option(False, "--daemon", "-d", help="后台运行，日志写入 logs/ingest.log"),
):
    """Ingest documents into the knowledge base."""
    if daemon:
        import subprocess
        import sys
        from pathlib import Path

        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "ingest.log"
        cmd = [sys.executable, "-m", "wagent.cli.main", "ingest", "--path", path, "--source", source]
        if verbose:
            cmd.append("--verbose")
        with open(log_file, "a") as f:
            proc = subprocess.Popen(
                cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True,
            )
        console.print(f"[green]Ingest 任务已启动[/green] (PID={proc.pid})")
        console.print(f"[dim]日志: {log_file.resolve()}[/dim]")
        console.print(f"[dim]查看进度: tail -f {log_file}[/dim]")
        return

    _setup_logging(verbose)
    console.print(f"[bold]Ingesting from {path} (source={source})[/bold]")
    from wagent.rag.ingest import run_ingest

    asyncio.run(run_ingest(path, source))


@app.command()
def update_kb(
    query: str = typer.Option("", help="Custom search query (optional)"),
):
    """Trigger knowledge base update via MCP crawl."""
    from wagent.mcp_servers.updater import KnowledgeUpdater

    console.print("[bold]Triggering knowledge base update...[/bold]")

    async def _run():
        updater = KnowledgeUpdater()
        report = await updater.run_update()
        console.print(f"[green]Update complete:[/green]")
        console.print(f"  Queries searched: {report['searched_queries']}")
        console.print(f"  Raw results: {report['raw_results']}")
        console.print(f"  Relevant extracted: {report['relevant_extracted']}")
        console.print(f"  Duplicates skipped: {report['duplicates_skipped']}")
        console.print(f"  Chunks added: {report['chunks_added']}")

    asyncio.run(_run())


@app.command()
def serve(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
    foreground: bool = typer.Option(False, "--foreground", "-f", help="前台运行，不守护进程化"),
):
    """Start the background scheduler for automated knowledge base updates.

    Runs APScheduler with CronTrigger every 6 hours.
    By default, the scheduler is daemonized (runs in background).
    Use --foreground to keep it in the current terminal.
    """
    if not foreground:
        import subprocess
        import sys
        from pathlib import Path

        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "serve.log"
        cmd = [sys.executable, "-m", "wagent.cli.main", "serve", "--foreground"]
        if verbose:
            cmd.append("--verbose")
        with open(log_file, "a") as f:
            proc = subprocess.Popen(
                cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True,
            )
        console.print(f"[green]WAgent 调度器已启动[/green] (PID={proc.pid})")
        console.print(f"[dim]日志: {log_file.resolve()}[/dim]")
        console.print("[dim]停止: wagent stop-serve[/dim]")
        (log_dir / "serve.pid").write_text(str(proc.pid))
        return

    _setup_logging(verbose)

    from wagent.config import get_settings
    cfg = get_settings()
    if cfg.qdrant_url:
        from wagent.cli.qdrant_docker import start_container, health_check
        try:
            start_container()
        except RuntimeError as e:
            console.print(f"[red]Qdrant 启动失败: {e}[/red]")
            raise typer.Exit(1)
        if not health_check():
            console.print("[red]Qdrant 健康检查未通过[/red]")
            raise typer.Exit(1)
        console.print("[green]Qdrant Docker 已就绪[/green]")

    from wagent.scheduler import run_scheduler

    console.print("[bold]Starting WAgent background scheduler...[/bold]")
    console.print("[dim]Knowledge base updates every 6 hours. Press Ctrl+C to stop.[/dim]")
    asyncio.run(run_scheduler())


@app.command(name="stop-serve")
def stop_serve():
    """Stop the background serve process gracefully."""
    import os
    import signal
    import time
    from pathlib import Path

    pid_file = Path("logs/serve.pid")
    if not pid_file.exists():
        console.print("[yellow]PID 文件不存在，后台调度器可能未运行。[/yellow]")
        raise typer.Exit(0)

    pid = int(pid_file.read_text().strip())
    try:
        os.kill(pid, 0)
    except OSError:
        console.print(f"[yellow]进程 {pid} 已不存在，清理 PID 文件。[/yellow]")
        pid_file.unlink(missing_ok=True)
        raise typer.Exit(0)

    console.print(f"[bold]正在停止后台调度器 (PID={pid})...[/bold]")
    os.kill(pid, signal.SIGTERM)

    for _ in range(20):
        time.sleep(0.5)
        try:
            os.kill(pid, 0)
        except OSError:
            break
    else:
        console.print("[red]进程未在 10 秒内退出，发送 SIGKILL...[/red]")
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass

    pid_file.unlink(missing_ok=True)
    console.print("[green]后台调度器已停止。[/green]")


@app.command()
def evaluate(
    profiles: str = typer.Option("expert,average,poor", help="Comma-separated candidate profiles"),
    runs: int = typer.Option(1, help="Runs per profile"),
    max_turns: int = typer.Option(8, help="Max turns per interview"),
):
    """Run automated interview evaluation with virtual candidates."""
    from wagent.evaluation.runner import EvaluationRunner

    console.print("[bold]Running automated evaluation...[/bold]")

    async def _run():
        runner = EvaluationRunner()
        profile_list = [p.strip() for p in profiles.split(",")]
        report = await runner.run_batch(
            profiles=profile_list,
            runs_per_profile=runs,
            max_turns=max_turns,
        )
        console.print(f"\n[bold green]Evaluation complete: {report['total_runs']} runs[/bold green]")
        for profile, data in report.get("per_profile", {}).items():
            console.print(
                f"  [{profile}] avg_score={data['avg_score']:.1f}, "
                f"coverage={data['avg_coverage']:.0%}, "
                f"recommendations={data['recommendations']}"
            )

    asyncio.run(_run())


@app.command()
def purge(
    source: str = typer.Option("crawled", "--source", "-s", help="Source type to delete (e.g. 'crawled')"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview only, do not delete"),
):
    """Delete knowledge base chunks by source type.

    Examples:
      wagent purge --source crawled           — delete all crawled data
      wagent purge --source crawled --dry-run  — preview without deleting
    """
    from wagent.rag.store import get_qdrant_client, count_by_source, delete_by_source, collection_stats

    client = get_qdrant_client()
    count = count_by_source(client, source)
    stats = collection_stats(client)
    total = stats["total_points"]

    console.print(
        f"[bold]知识库状态[/bold]: {total} chunks 总计, "
        f"其中 source='{source}' 有 [bold red]{count}[/bold red] chunks"
    )

    if count == 0:
        console.print(f"[green]没有 source='{source}' 的数据需要删除[/green]")
        return

    if dry_run:
        console.print(f"[yellow][DRY RUN] 将删除 {count} chunks (source='{source}'), 剩余 {total - count} chunks[/yellow]")
        return

    confirmed = Confirm.ask(
        f"确认删除 {count} chunks (source='{source}')？删除后剩余 {total - count} chunks",
        default=False,
    )
    if not confirmed:
        console.print("[yellow]已取消[/yellow]")
        return

    deleted = delete_by_source(client, source)
    remaining = collection_stats(client)["total_points"]
    console.print(f"[green]已删除 {deleted} chunks[/green], 剩余 {remaining} chunks")
    console.print("[dim]提示: 下次启动 interview/qa 时 BM25 索引会自动重建[/dim]")


@app.command()
def prepare(
    resume: str = typer.Argument(..., help="Path to candidate resume PDF"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
):
    """Prepare the knowledge base for a specific resume.

    Full pipeline: parse resume -> extract skills -> GAP analysis ->
    search & fill gaps -> topic review -> readiness report.

    Examples:
      wagent prepare resume.pdf
    """
    _setup_logging(verbose)

    from pathlib import Path

    if not Path(resume).exists():
        console.print(f"[red]文件不存在: {resume}[/red]")
        raise typer.Exit(1)

    async def _run():
        from rich.table import Table
        from wagent.cli.interview_session import parse_resume_pdf
        from wagent.agents.router import extract_skills
        from wagent.mcp_servers.updater import KnowledgeUpdater
        from wagent.config import load_topic_taxonomy, save_topic_taxonomy
        from wagent.rag.store import get_qdrant_client, collection_stats, COLLECTION_NAME

        # ── 1. Parse resume ──
        console.print("\n[bold]== Step 1/5: 解析简历 ==[/bold]")
        resume_text = parse_resume_pdf(resume)
        if not resume_text.strip():
            console.print("[red]简历解析失败或内容为空[/red]")
            return
        console.print(f"[green]简历解析完成[/green] ({len(resume_text)} chars)")

        # ── 2. Extract skills & map topics ──
        console.print("\n[bold]== Step 2/5: 提取技能 + 话题映射 ==[/bold]")
        extraction = await extract_skills(resume_text)
        console.print(f"  候选人: {extraction.candidate_name}")
        console.print(f"  技能数: {len(extraction.skills)}")
        console.print(f"  项目关键词: {extraction.project_keywords[:8]}")
        console.print(f"  映射话题: {extraction.mapped_topics}")

        # ── 3. GAP analysis ──
        console.print("\n[bold]== Step 3/5: 知识库 GAP 分析 ==[/bold]")
        updater = KnowledgeUpdater()
        weak_topics = await updater.resume_gap_analysis(
            extraction.mapped_topics, chunk_threshold=5
        )

        if weak_topics:
            console.print("[yellow]发现以下话题在知识库中覆盖不足：[/yellow]")
            for wt in weak_topics:
                topic = wt.get("topic", "")
                count = wt.get("current_count", 0)
                keywords = wt.get("search_keywords", [])
                console.print(f"  • {topic} ({count} chunks) → 搜索词: {keywords}")

            # ── 4. Fill gaps ──
            console.print("\n[bold]== Step 4/5: 搜索补全知识库 ==[/bold]")
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
                console.print("[dim]已跳过搜索补全[/dim]")
            else:
                selected_topics = decision["params"].get("items", []) if decision["action"] == "fill_selected" else None

                all_keywords = []
                for wt in weak_topics:
                    if selected_topics is None or wt.get("topic", "") in selected_topics:
                        all_keywords.extend(wt.get("search_keywords", []))
                if all_keywords:
                    if selected_topics:
                        console.print(f"[dim]仅搜索: {selected_topics}[/dim]")
                    console.print(f"[dim]搜索关键词 ({len(all_keywords)}): {all_keywords[:6]}...[/dim]")
                    fill_report = await updater.fill_gaps(all_keywords)
                    console.print(
                        f"[green]补全完成[/green]: 搜索 {fill_report['searched']} 条, "
                        f"提取 {fill_report['extracted']} 条, 入库 {fill_report['added']} 条"
                    )
                else:
                    console.print("[dim]没有可用的搜索关键词[/dim]")
        else:
            console.print("[green]知识库对简历相关话题覆盖良好，无需补全[/green]")
            console.print("\n[bold]== Step 4/5: 搜索补全（跳过）==[/bold]")

        # ── 5. Topic taxonomy review ──
        console.print("\n[bold]== Step 5/5: Topic 自动审查 ==[/bold]")
        review = await updater.review_taxonomy(auto_accept=False)
        console.print(
            f"  知识库: {review['total_chunks']} chunks, "
            f"general={review['general_count']} ({review['general_ratio']:.1%})"
        )

        if review["proposed_topics"]:
            console.print(f"\n[bold]LLM 提议新增 topic:[/bold]")
            for t in review["proposed_topics"]:
                console.print(f"  • {t}")
            console.print(f"[dim]原因: {review['reason']}[/dim]\n")

            topic_decision = await prompt_and_parse(
                "[bold green]确认添加这些 topic？[/bold green]（可输入：全部添加 / 只加XX和YY / 不添加）",
                situation=f"LLM 提议新增以下 topic: {review['proposed_topics']}。用户需要决定是否添加。",
                actions=[
                    {"id": "accept_all", "desc": "全部添加"},
                    {"id": "accept_selected", "desc": "只添加用户指定的 topic", "has_items": True},
                    {"id": "reject", "desc": "不添加任何 topic"},
                ],
                available_items=review["proposed_topics"],
            )

            if topic_decision["action"] == "reject":
                console.print("[dim]已跳过 topic 添加[/dim]")
            else:
                topics_to_add = (
                    topic_decision["params"].get("items", review["proposed_topics"])
                    if topic_decision["action"] == "accept_selected"
                    else review["proposed_topics"]
                )
                if not topics_to_add:
                    topics_to_add = review["proposed_topics"]
                new_taxonomy = load_topic_taxonomy() + topics_to_add
                save_topic_taxonomy(new_taxonomy)
                console.print(f"[green]已保存 {len(topics_to_add)} 个新 topic[/green]")
                console.print("[dim]正在重分类 general chunks...[/dim]")
                reclass = await updater.reclassify_general_chunks()
                console.print(
                    f"  重分类完成: reclassified={reclass['reclassified']}, "
                    f"still_general={reclass['still_general']}"
                )
        else:
            console.print(f"[dim]{review['reason']}[/dim]")

        # ── Readiness report ──
        console.print("\n" + "=" * 60)
        console.print("[bold green]准备完成！知识库就绪报告[/bold green]\n")

        taxonomy = load_topic_taxonomy()
        client = get_qdrant_client()
        topic_counts: dict[str, int] = {t: 0 for t in taxonomy}
        topic_counts["general"] = 0
        source_counts: dict[str, int] = {}
        total = 0

        offset = None
        while True:
            points, next_offset = client.scroll(
                collection_name=COLLECTION_NAME, limit=500, offset=offset, with_payload=True,
            )
            for point in points:
                total += 1
                topic = point.payload.get("topic", "general")
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
                src = point.payload.get("source", "unknown")
                source_counts[src] = source_counts.get(src, 0) + 1
            if next_offset is None:
                break
            offset = next_offset

        table = Table(title="知识库概况")
        table.add_column("Topic", style="cyan")
        table.add_column("Chunks", justify="right", style="green")
        table.add_column("简历相关", justify="center")

        for topic in taxonomy + ["general"]:
            count = topic_counts.get(topic, 0)
            is_resume = "✓" if topic in extraction.mapped_topics else ""
            table.add_row(topic, str(count), is_resume)
        table.add_section()
        table.add_row("[bold]Total[/bold]", f"[bold]{total}[/bold]", "")

        console.print(table)

        console.print("\n[bold]数据来源分布:[/bold]")
        for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
            console.print(f"  {src}: {cnt}")

        resume_coverage = sum(1 for t in extraction.mapped_topics if topic_counts.get(t, 0) >= 5)
        resume_total = len(extraction.mapped_topics)
        console.print(
            f"\n[bold]简历话题覆盖: {resume_coverage}/{resume_total}[/bold] "
            f"(≥5 chunks 视为已覆盖)"
        )

        console.print("\n[bold]下一步:[/bold]")
        console.print(f"  • 知识问答:   [cyan]wagent qa[/cyan]")
        console.print(f"  • 模拟面试:   [cyan]wagent interview {resume}[/cyan]")
        console.print(f"  • 查看 topic: [cyan]wagent topics[/cyan]")

    asyncio.run(_run())


@app.command()
def topics(
    review: bool = typer.Option(False, "--review", "-r", help="Trigger LLM taxonomy review"),
    add: str = typer.Option("", "--add", "-a", help="Manually add a new topic (snake_case)"),
    reclassify: bool = typer.Option(False, "--reclassify", help="Re-classify all 'general' chunks"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
):
    """View, review, and manage the topic taxonomy.

    Examples:
      wagent topics            — show current topics + distribution
      wagent topics --review   — LLM proposes new topics from 'general' chunks
      wagent topics --add "knowledge_graph"  — manually add a topic & reclassify
      wagent topics --reclassify             — re-classify 'general' chunks only
    """
    _setup_logging(verbose)

    from wagent.config import load_topic_taxonomy, save_topic_taxonomy

    async def _run():
        taxonomy = load_topic_taxonomy()

        # --- Manual add ---
        if add:
            topic_name = add.strip().lower().replace(" ", "_")
            if topic_name in taxonomy:
                console.print(f"[yellow]Topic '{topic_name}' 已存在[/yellow]")
            else:
                taxonomy.append(topic_name)
                save_topic_taxonomy(taxonomy)
                console.print(f"[green]已添加 topic: '{topic_name}'[/green]")
                console.print("[dim]开始重分类 general chunks...[/dim]")
                from wagent.mcp_servers.updater import KnowledgeUpdater
                updater = KnowledgeUpdater()
                stats = await updater.reclassify_general_chunks()
                console.print(
                    f"  重分类完成: reclassified={stats['reclassified']}, "
                    f"still_general={stats['still_general']}, errors={stats['errors']}"
                )
            return

        # --- LLM Review ---
        if review:
            from wagent.mcp_servers.updater import KnowledgeUpdater
            console.print("[bold]正在审查 topic 体系...[/bold]")
            updater = KnowledgeUpdater()
            result = await updater.review_taxonomy(auto_accept=False)

            console.print(
                f"\n知识库: {result['total_chunks']} chunks, "
                f"general={result['general_count']} ({result['general_ratio']:.1%})"
            )

            if not result["proposed_topics"]:
                console.print(f"[dim]{result['reason']}[/dim]")
                return

            console.print(f"\n[bold]LLM 提议新增 topic:[/bold]")
            for t in result["proposed_topics"]:
                console.print(f"  • {t}")
            console.print(f"[dim]原因: {result['reason']}[/dim]\n")

            from wagent.cli.smart_prompt import prompt_and_parse
            topic_decision = await prompt_and_parse(
                "[bold green]确认添加这些 topic？[/bold green]（可输入：全部添加 / 只加XX和YY / 不添加）",
                situation=f"LLM 提议新增以下 topic: {result['proposed_topics']}。用户需要决定是否添加。",
                actions=[
                    {"id": "accept_all", "desc": "全部添加"},
                    {"id": "accept_selected", "desc": "只添加用户指定的 topic", "has_items": True},
                    {"id": "reject", "desc": "不添加任何 topic"},
                ],
                available_items=result["proposed_topics"],
            )

            if topic_decision["action"] == "reject":
                console.print("[yellow]已跳过[/yellow]")
            else:
                topics_to_add = (
                    topic_decision["params"].get("items", result["proposed_topics"])
                    if topic_decision["action"] == "accept_selected"
                    else result["proposed_topics"]
                )
                if not topics_to_add:
                    topics_to_add = result["proposed_topics"]
                new_taxonomy = load_topic_taxonomy() + topics_to_add
                save_topic_taxonomy(new_taxonomy)
                console.print(f"[green]已保存 {len(topics_to_add)} 个新 topic[/green]")
                console.print("[dim]开始重分类 general chunks...[/dim]")
                stats = await updater.reclassify_general_chunks()
                console.print(
                    f"  重分类完成: reclassified={stats['reclassified']}, "
                    f"still_general={stats['still_general']}, errors={stats['errors']}"
                )
            return

        # --- Reclassify only ---
        if reclassify:
            from wagent.mcp_servers.updater import KnowledgeUpdater
            console.print("[bold]开始重分类 general chunks...[/bold]")
            updater = KnowledgeUpdater()
            stats = await updater.reclassify_general_chunks()
            console.print(
                f"重分类完成: total={stats['total_general']}, "
                f"reclassified={stats['reclassified']}, "
                f"still_general={stats['still_general']}, errors={stats['errors']}"
            )
            return

        # --- Default: show current taxonomy + distribution ---
        from wagent.rag.store import get_qdrant_client
        from rich.table import Table

        console.print(f"[bold]Topic Taxonomy ({len(taxonomy)} topics)[/bold]\n")

        client = get_qdrant_client()
        topic_counts: dict[str, int] = {t: 0 for t in taxonomy}
        topic_counts["general"] = 0
        total = 0

        try:
            offset = None
            while True:
                points, next_offset = client.scroll(
                    collection_name="knowledge_base",
                    limit=500,
                    offset=offset,
                    with_payload=True,
                )
                for point in points:
                    total += 1
                    topic = point.payload.get("topic", "general")
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1
                if next_offset is None:
                    break
                offset = next_offset
        except Exception:
            pass

        table = Table(title="Topic 分布")
        table.add_column("Topic", style="cyan")
        table.add_column("Chunks", justify="right", style="green")
        table.add_column("占比", justify="right")

        for topic in taxonomy + ["general"]:
            count = topic_counts.get(topic, 0)
            pct = f"{count / total:.1%}" if total else "–"
            style = "bold red" if topic == "general" and total and count / total > 0.15 else ""
            table.add_row(topic, str(count), pct, style=style)

        table.add_section()
        table.add_row("[bold]Total[/bold]", f"[bold]{total}[/bold]", "100%")

        console.print(table)
        console.print(
            f"\n[dim]配置文件: data/topic_taxonomy.json | "
            f"使用 --review 审查 | --add 手动添加 | --reclassify 重分类[/dim]"
        )

    asyncio.run(_run())


@app.command()
def qdrant(
    action: str = typer.Argument("status", help="Action: status / start / stop / remove"),
):
    """Manage the Qdrant Docker container.

    Examples:
      wagent qdrant            — show container status
      wagent qdrant start      — start the container
      wagent qdrant stop       — stop the container
      wagent qdrant remove     — stop and remove the container
    """
    from wagent.cli.qdrant_docker import (
        start_container, stop_container, remove_container, status_info,
    )

    if action == "status":
        info = status_info()
        console.print(f"[bold]Container:[/bold] {info['container']}")
        console.print(f"[bold]Running:[/bold]   {info['running']}")
        console.print(f"[bold]Healthy:[/bold]   {info['healthy']}")
    elif action == "start":
        start_container()
        console.print("[green]Qdrant container started[/green]")
    elif action == "stop":
        stop_container()
        console.print("[yellow]Qdrant container stopped[/yellow]")
    elif action == "remove":
        remove_container()
        console.print("[yellow]Qdrant container removed[/yellow]")
    else:
        console.print(f"[red]Unknown action: {action}. Use: status / start / stop / remove[/red]")


if __name__ == "__main__":
    app()
