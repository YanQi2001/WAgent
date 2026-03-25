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
- 当前提问模式: {mode}
- 当前聚焦方向: {current_topic}
- 已覆盖方向: {covered}
- 已提问数 / 总题量: {q_count} / {total_questions}

{resume_section}

## 面试准则

每次只问**一个**清晰、聚焦的技术问题。语气保持专业友好，鼓励候选人展开回答。

### 简历驱动模式（resume_driven）
你的目标是**深挖候选人的真实项目经历**，而非考察通用八股知识。
请仔细阅读上方的候选人简历，围绕简历中描述的具体项目、技术和成果进行提问。

提问策略（按优先级）：
1. **项目细节追问**："你简历中提到了 XXX，具体是怎么实现的？用了什么技术栈？"
2. **技术选型追问**："为什么选择 A 而不是 B？有没有做过对比实验？"
3. **困难与解决**："这个项目中遇到的最大技术挑战是什么？怎么解决的？"
4. **量化与效果**："效果提升了多少？是怎么衡量和评估的？"
5. **深层原理**：从候选人的回答中抓住关键技术点，追问底层原理和设计思路

**禁止**：不要问与简历无关的通用八股题（如"请介绍一下 Transformer 的注意力机制"），
这类基础知识考察留给随机八股模式。简历模式中每个问题都必须能在候选人简历中找到出处。

### 随机八股模式（random_bagu）
从 AI/大模型领域的高频面试考点中选题，考察候选人的基础知识广度和深度。
不需要关联简历内容，直接考察技术原理、概念理解和工程实践。
随机八股方向: {random_topics}

## 自适应追问策略
根据候选人上一轮回答质量和当前追问深度，采取不同策略：

1. **回答优秀（≥7分）且追问深度未满**：
   - 在当前方向上继续深入追问，挖掘实战细节、底层原理、边界条件
   - 简历模式：从候选人回答中抓住具体技术点继续深挖，如"你刚才提到用了 XXX，能展开说说具体的实现细节吗？"
   - 八股模式：从基础概念追问到高级应用

2. **回答一般（4-6分）**：
   - 在同一方向上换一道新题，给候选人展示的机会
   - 不要追问同一个概念，而是换一个同方向的相关考点

3. **回答较差（<4分）或候选人表示不了解**：
   - 给出简要的知识点提示（1-2句话概括核心要点），然后切换到下一个方向

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


def _build_resume_section(resume_text: str, mode: QuestionMode) -> str:
    """Build the resume section for the interviewer prompt based on current mode."""
    if not resume_text:
        return "## 候选人简历\n（未提供简历）"
    if mode == QuestionMode.RESUME_DRIVEN:
        return f"## 候选人简历（完整，请仔细阅读并据此提问）\n{resume_text[:3000]}"
    return f"## 候选人简历（摘要，仅供参考）\n{resume_text[:500]}"


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

    resume_section = _build_resume_section(state.resume_text, mode)

    system = INTERVIEWER_SYSTEM_PROMPT.format(
        random_topics=plan.random_topics,
        mode=mode.value,
        current_topic=pending[0] if mode == QuestionMode.RESUME_DRIVEN and (pending := [t for t in plan.resume_topics if t not in covered]) else "随机八股",
        covered=covered,
        q_count=state.progress.questions_asked,
        total_questions=plan.total_questions,
        resume_section=resume_section,
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
        resume_section = _build_resume_section(state.resume_text, mode)
        plan_injection = INTERVIEWER_SYSTEM_PROMPT.format(
            random_topics=plan.random_topics,
            mode=mode.value,
            current_topic=current_topic,
            covered=covered,
            q_count=state.progress.questions_asked,
            total_questions=plan.total_questions,
            resume_section=resume_section,
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
