"""Virtual candidate – simulates different candidate profiles for automated evaluation.

Profiles:
  - "expert": Deep knowledge, articulate answers
  - "average": Reasonable understanding, some gaps
  - "poor": Superficial answers, off-topic tendencies
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from wagent.llm import get_llm

logger = logging.getLogger(__name__)

CANDIDATE_PROFILES = {
    "expert": """\
你是一位拥有 5 年以上经验的资深 AI 工程师。你对 Transformer、RAG 系统、大模型微调和 Agent 架构有深入的理解。
回答面试问题时：
- 提供详尽、技术准确的回答
- 引用具体的论文、框架和技术方案
- 解释技术权衡和设计决策
- 在适当的时候提出澄清性问题
使用中文回答。""",

    "average": """\
你是一位有 1-2 年经验的初中级 AI 工程师。你对基础知识有合理的理解，但在高级话题上存在知识空白。
回答面试问题时：
- 给出大致正确但有时浅显的回答
- 偶尔混淆相关概念
- 面对深入追问时表现吃力
- 不知道的地方坦诚承认
使用中文回答。""",

    "poor": """\
你是一位刚完成在线 AI 课程的初学者。
回答面试问题时：
- 给出模糊、表面化的回答
- 有时会误解问题
- 依赖流行词但缺乏真正的理解
- 偶尔跑题
使用中文回答。""",
}


class VirtualCandidate:
    """Simulates a candidate for automated interview evaluation."""

    def __init__(self, profile: str = "average"):
        if profile not in CANDIDATE_PROFILES:
            raise ValueError(f"Unknown profile: {profile}. Use: {list(CANDIDATE_PROFILES.keys())}")
        self.profile = profile
        self._system_prompt = CANDIDATE_PROFILES[profile]

    async def answer(self, question: str, context: str = "") -> str:
        """Generate a candidate answer to an interview question."""
        llm = get_llm(temperature=0.6, max_tokens=500)

        messages = [
            SystemMessage(content=self._system_prompt),
        ]
        if context:
            messages.append(SystemMessage(content=f"面试上下文:\n{context}"))
        messages.append(HumanMessage(content=question))

        response = await llm.ainvoke(messages)
        return response.content
