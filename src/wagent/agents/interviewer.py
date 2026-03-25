"""Interviewer Agent – conducts the actual interview with adaptive follow-up.

Operates in ReAct mode: decides whether to search the knowledge base,
formulates questions, evaluates candidate answers, and determines follow-up depth.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from wagent.agents.schemas import AnswerEvaluation, InterviewPlan
from wagent.harness.state import InterviewState, QAPair, QuestionMode
from wagent.llm import get_llm
from wagent.utils import strip_json_fences

logger = logging.getLogger(__name__)

INTERVIEWER_SYSTEM_PROMPT = """\
你是一位资深互联网大厂（字节跳动/阿里巴巴/腾讯级别）的 AI 技术面试官。
你精通大模型、推荐系统、NLP、Agent 架构等 AI 全栈技术，面试风格专业、深入但不刻板。

## 面试计划
- 简历驱动方向: {resume_topics}
- 随机八股方向: {random_topics}
- 当前提问模式: {mode}
- 当前聚焦方向: {current_topic}
- 已覆盖方向: {covered}
- 已提问数 / 总题量: {q_count} / {total_questions}

## 面试准则
- 每次只问**一个**清晰、聚焦的技术问题
- 简历驱动模式：紧扣候选人简历中描述的项目经历和技术栈进行追问
- 随机八股模式：从 AI/大模型领域的高频面试考点中选题
- 语气保持专业友好，鼓励候选人展开回答

## 自适应追问策略
根据候选人上一轮回答质量和当前追问深度，采取不同策略：

1. **回答优秀（≥7分）且追问深度未满**：
   - 在当前方向上继续深入追问，挖掘实战细节、底层原理、边界条件
   - 例如从 "你用了什么方法" 追问到 "为什么选择这个方法" 再到 "遇到了什么坑、怎么解决的"

2. **回答一般（4-6分）**：
   - 在同一方向上换一道新题，给候选人展示的机会
   - 不要追问同一个概念，而是换一个同方向的相关考点

3. **回答较差（<4分）或候选人表示不了解**：
   - 给出简要的知识点提示（1-2句话概括核心要点），然后切换到下一个方向
   - 知识点提示格式：「💡 简要提示：...」

4. **追问深度已满（连续追问达上限）**：
   - 对当前方向做简要总结，自然过渡到下一个方向

{knowledge_context}

请提出你的下一个面试问题。
"""

EVAL_PROMPT = """\
评估候选人对 AI 技术面试问题的回答。

## 问题
{question}

## 候选人回答
{answer}

## 所属方向
{topic}

## 评估要求
请从 0-10 分打分，并判断以下维度：
- depth: "basic"（基础）| "intermediate"（中级）| "advanced"（高级）
- should_follow_up: 是否值得在当前方向继续追问
  * score >= 7: 强烈建议追问（挖掘深度）
  * 4 <= score < 7: 建议追问（换同方向新题）
  * score < 4: 不建议追问（应切换方向）
- follow_up_direction: 如果建议追问，下一步追问的具体方向
- notes: 简要评价要点（理解深度、实战经验、表达清晰度）
- is_exit_request: 用户输入是否是在请求退出/结束面试（而非回答技术问题）
  * 如果用户表达了"退出面试""面试结束""不想继续了""结束吧"等意图 → true
  * 如果用户在尝试回答技术问题（哪怕回答很差或说"不知道"）→ false

以 JSON 格式回复:
{{"score": 7.0, "depth": "intermediate", "should_follow_up": true, "follow_up_direction": "追问方向...", "notes": "评价...", "is_exit_request": false}}
"""


async def generate_question(
    state: InterviewState,
    plan: InterviewPlan,
    knowledge_context: str = "",
) -> str:
    """Generate the next interview question based on current state and plan."""
    mode = state.progress.current_mode
    covered = state.progress.covered_topics

    if mode == QuestionMode.RESUME_DRIVEN:
        pending = [t for t in plan.resume_topics if t not in covered]
        topic_hint = f"请优先覆盖以下尚未提问的简历方向: {pending}" if pending else ""
    else:
        topic_hint = f"请从以下随机八股方向中选题: {plan.random_topics}"

    kb_section = ""
    if knowledge_context:
        kb_section = f"相关知识库参考资料:\n{knowledge_context}\n\n请参考以上内容来设计你的提问。"

    system = INTERVIEWER_SYSTEM_PROMPT.format(
        resume_topics=plan.resume_topics,
        random_topics=plan.random_topics,
        mode=mode.value,
        current_topic=pending[0] if mode == QuestionMode.RESUME_DRIVEN and (pending := [t for t in plan.resume_topics if t not in covered]) else "随机八股",
        covered=covered,
        q_count=state.progress.questions_asked,
        total_questions=plan.total_questions,
        knowledge_context=kb_section,
    )

    messages: list[BaseMessage] = [SystemMessage(content=system)]

    for qa in state.qa_history[-4:]:
        messages.append(AIMessage(content=f"[Previous Q] {qa.question}"))
        messages.append(HumanMessage(content=f"[Previous A] {qa.answer}"))

    if topic_hint:
        messages.append(SystemMessage(content=f"[Harness 调度指令] {topic_hint}"))

    messages.append(HumanMessage(content="请提出下一个面试问题。"))

    llm = get_llm(temperature=0.5, max_tokens=8192)
    response = await llm.ainvoke(messages)
    return response.content


async def evaluate_answer(
    question: str,
    answer: str,
    topic: str,
) -> AnswerEvaluation:
    """Evaluate a candidate's answer and decide on follow-up."""
    llm = get_llm(tier="fast", temperature=0.1, max_tokens=500)
    prompt = EVAL_PROMPT.format(question=question, answer=answer, topic=topic)
    response = await llm.ainvoke([
        SystemMessage(content="你是一个精确的 JSON 评估助手，只输出合法的 JSON，不要输出任何其他内容。"),
        HumanMessage(content=prompt),
    ])

    try:
        data = json.loads(strip_json_fences(response.content))
        return AnswerEvaluation(**data)
    except (json.JSONDecodeError, Exception) as e:
        logger.error("Failed to parse answer evaluation: %s", e)
        return AnswerEvaluation(score=5.0, depth="basic", should_follow_up=False)


async def interviewer_turn(
    state: InterviewState,
    messages: list[BaseMessage],
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Agent callable for the harness: take state + messages + context, return response dict.

    context may contain:
      - plan: InterviewPlan
      - retriever: HybridRetriever instance
      - current_topic: str
    """
    context = context or {}
    plan: InterviewPlan | None = context.get("plan")
    retriever = context.get("retriever")
    current_topic: str = context.get("current_topic", "general")

    kb_section = ""
    if retriever and current_topic:
        try:
            results = retriever.retrieve(current_topic, topic_filter=None)
            if results:
                snippets = [r["text"][:300] for r in results[:3]]
                kb_section = "相关知识库内容:\n" + "\n---\n".join(snippets)
        except Exception as e:
            logger.debug("RAG retrieval in interviewer_turn failed: %s", e)

    if plan:
        mode = state.progress.current_mode
        covered = state.progress.covered_topics
        plan_injection = INTERVIEWER_SYSTEM_PROMPT.format(
            resume_topics=plan.resume_topics,
            random_topics=plan.random_topics,
            mode=mode.value,
            current_topic=current_topic,
            covered=covered,
            q_count=state.progress.questions_asked,
            total_questions=plan.total_questions,
            knowledge_context=kb_section,
        )
        messages = [SystemMessage(content=plan_injection)] + [
            m for m in messages if not isinstance(m, SystemMessage) or m.content != messages[0].content
        ]
    elif kb_section:
        messages.append(SystemMessage(content=f"[知识库上下文]\n{kb_section}"))

    llm = get_llm(temperature=0.4, max_tokens=8192)
    response = await llm.ainvoke(messages)

    input_tokens = response.response_metadata.get("token_usage", {}).get("prompt_tokens", 0)
    output_tokens = response.response_metadata.get("token_usage", {}).get("completion_tokens", 0)

    return {
        "response": response.content,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
