"""Judge Agent – asynchronous scoring, decoupled from the Interviewer.

Implements "examiner and grader separation" pattern with LLM-as-Judge.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from wagent.agents.schemas import InterviewScorecard, StudyItem, TopicScoreItem
from wagent.harness.state import InterviewState
from wagent.llm import get_llm
from wagent.utils import strip_json_fences

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """\
你是一位独立公正的 AI 技术面试评委。你独立于面试官，根据面试记录对候选人进行客观评价。

## 核心评价维度

1. **各方向技术深度** (0-10): 对每个考察过的方向分别打分。
2. **综合技术能力** (0-10): 整体评估候选人的技术水平。
3. **优势与不足**: 引用具体的问答编号作为佐证。
4. **录用建议**: "强烈推荐录用" | "推荐录用" | "待定" | "倾向不录用" | "不推荐"

## 高级评价维度

5. **实战经验指数** (0-10): 区分真实上手经验与教科书式背诵。如果候选人说"我们在 batch_size=64 时遇到 OOM，于是采用了梯度检查点"得分应高于仅说"梯度检查点可以降低显存占用"。关注：具体的故障排查经历、线上调试故事、资源约束下的权衡决策、真实系统的数据指标。

6. **第一性原理思维** (0-10): 候选人是否从底层机制推导答案，而非仅仅回忆记忆的模式？面对新情境能否组合基础概念进行推理？是否解释了"为什么"，而不只是"是什么"？

7. **STAR完整度** (0-10): 对候选人描述的 1-2 个核心项目经历，评估其完整度：Situation（场景与约束）、Task（具体职责）、Action（采取的具体步骤、使用的工具、做出的决策）、Result（可量化的成果、经验总结）。缺失或模糊的部分要扣分。

8. **追问韧性** (0-10): 当面试官深入追问时，候选人的回答质量是提升（说明有深度）、持平（触及知识边界）、还是下降（暴露浅层理解）？这是衡量真实掌握程度的最强信号之一。

## 薄弱知识点学习建议

9. **study_guide**: 基于面试中暴露的薄弱环节，为候选人生成针对性的学习建议列表。每条建议包含：
   - topic: 薄弱方向名称
   - weakness_summary: 具体哪里不熟（引用问答编号佐证）
   - suggested_keywords: 建议候选人搜索学习的关键词（3-5 个）
   - priority: "high"（得分 <4 或完全不会）| "medium"（得分 4-6）| "low"（有一定基础但不够深入）

请做到客观公正、以证据为基础，引用具体的问答编号来支撑你的评价。
输出合法的 JSON。
"""

JUDGE_PROMPT = """\
候选人: {candidate_name}
考察方向: {resume_topics}

## 面试问答记录
{qa_text}

请生成评分报告（JSON 格式）:
{{
  "overall_score": 7.5,
  "topic_scores": [{{"topic": "方向名", "score": 8.0, "notes": "评价..."}}],
  "strengths": ["优势1", "优势2"],
  "weaknesses": ["不足1", "不足2"],
  "recommendation": "推荐录用",
  "summary": "整体评价概述...",
  "battle_scars_index": 6.0,
  "first_principles_score": 7.0,
  "star_completeness": 5.5,
  "followup_resilience": 7.0,
  "deep_analysis": "1-2 段深入分析：候选人表现最好的环节使用了什么方法论，展现了什么技术偏好，暴露了什么技术边界。",
  "study_guide": [
    {{"topic": "薄弱方向", "weakness_summary": "具体哪里不熟（引用 Q 编号）", "suggested_keywords": ["关键词1", "关键词2", "关键词3"], "priority": "high"}},
    {{"topic": "另一薄弱方向", "weakness_summary": "具体表现", "suggested_keywords": ["关键词"], "priority": "medium"}}
  ]
}}
"""


async def judge_interview(state: InterviewState) -> InterviewScorecard:
    """Run the Judge agent on the completed interview."""
    qa_lines = []
    for i, qa in enumerate(state.qa_history, 1):
        qa_lines.append(f"Q{i} [{qa.topic}]: {qa.question}")
        qa_lines.append(f"A{i}: {qa.answer}")
        if qa.score is not None:
            qa_lines.append(f"  (Interviewer score: {qa.score}/10)")
        qa_lines.append("")

    prompt = JUDGE_PROMPT.format(
        candidate_name=state.candidate_name or "Unknown",
        resume_topics=state.resume_topics,
        qa_text="\n".join(qa_lines),
    )

    llm = get_llm(temperature=0.1, max_tokens=8192)
    response = await llm.ainvoke([
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    try:
        data = json.loads(strip_json_fences(response.content))
        return InterviewScorecard(**data)
    except (json.JSONDecodeError, Exception) as e:
        logger.error("Failed to parse judge scorecard: %s", e)
        return InterviewScorecard(
            overall_score=0,
            summary=f"Judge parsing failed: {e}. Raw: {response.content[:500]}",
            recommendation="error",
        )
