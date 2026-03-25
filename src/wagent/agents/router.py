"""Router Agent – parses resume, extracts skills, generates interview plan.

Uses a lightweight LLM call with structured output to map resume content
to the topic taxonomy, then produces a question plan (7:3 split).
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from wagent.agents.schemas import InterviewPlan, SkillExtraction
from wagent.harness.state import InterviewState, SkillEntity
from wagent.config import load_topic_taxonomy, save_topic_taxonomy
from wagent.llm import get_llm
from wagent.utils import strip_json_fences

logger = logging.getLogger(__name__)

EXTRACT_PROMPT = """\
你是一位 AI 面试准备助手，负责分析候选人简历并提取结构化信息。

## 任务

请按照以下步骤分析简历：

1. **提取基本信息**：候选人姓名、技术技能（含分类和熟练度）、项目关键词
2. **自由提取技术方向**（free_topics）：不受任何限制，用**中文短语**列出候选人涉及的所有技术方向。贴近中国互联网企业面试的叫法，例如"推荐系统"、"大模型训练"、"分布式系统"、"多模态模型"等
3. **对比映射**：将 free_topics 与下方已有 taxonomy 做宽松匹配（有交集即匹配），输出 mapped_topics
4. **识别未覆盖方向**：将无法映射到 taxonomy 的方向放入 unmapped_topics（供后续 taxonomy 自动扩展）
5. **建议提问数量**：根据简历技术深度建议 15-20 题

## 已有 topic taxonomy
{taxonomy}

## 输出格式（严格 JSON）
{{
  "candidate_name": "姓名",
  "skills": [{{"name": "技能名", "category": "分类", "proficiency": "熟练度"}}],
  "project_keywords": ["关键词1", "关键词2"],
  "free_topics": ["自由提取的方向1", "自由提取的方向2"],
  "mapped_topics": ["匹配到taxonomy的方向"],
  "unmapped_topics": ["未匹配的方向"],
  "suggested_question_count": 15
}}

## 候选人简历
{resume_text}
"""

PLAN_PROMPT = """\
根据简历技能提取结果，制定面试计划。

## 技能提取结果
{extraction}

## 规则
- resume_topics：从 free_topics 和 mapped_topics 中选取与候选人背景匹配的方向（面试模式 A：简历驱动）
- random_topics：从 taxonomy 中选取候选人简历未涉及的方向（面试模式 B：随机八股）
- resume_question_count：约占总题量的 80%，每个 resume topic 至少 1 题，侧重项目深挖
- random_question_count：约占总题量的 20%。若简历涉及方向少（<4个），适当增加随机题量
- total_questions：15-20 题

## 输出格式（严格 JSON）
{{
  "resume_topics": ["方向1", "方向2"],
  "random_topics": ["方向3", "方向4"],
  "resume_question_count": 10,
  "random_question_count": 5,
  "total_questions": 15
}}
"""


async def extract_skills(resume_text: str) -> SkillExtraction:
    """Extract skills and map to topic taxonomy."""
    taxonomy = load_topic_taxonomy()
    llm = get_llm(temperature=0.1, max_tokens=8192)
    prompt = EXTRACT_PROMPT.format(
        taxonomy=json.dumps(taxonomy, ensure_ascii=False),
        resume_text=resume_text[:5000],
    )
    response = await llm.ainvoke([
        SystemMessage(content="你是一个精确的 JSON 提取助手，只输出合法的 JSON，不要输出任何其他内容。"),
        HumanMessage(content=prompt),
    ])

    raw = response.content or ""
    cleaned = strip_json_fences(raw)
    logger.debug("extract_skills raw response (%d chars): %.300s", len(raw), raw)

    try:
        data = json.loads(cleaned)
        extraction = SkillExtraction(**data)

        unmapped = extraction.unmapped_topics
        if unmapped:
            current_taxonomy = load_topic_taxonomy(force_reload=True)
            new_topics = [t for t in unmapped if t not in current_taxonomy]
            if new_topics:
                save_topic_taxonomy(current_taxonomy + new_topics)
                logger.info("extract_skills: 自动扩展 taxonomy，新增 %d 个 topic: %s", len(new_topics), new_topics)
                extraction.mapped_topics = list(extraction.mapped_topics) + new_topics

        return extraction
    except (json.JSONDecodeError, Exception) as e:
        logger.error("Failed to parse skill extraction: %s | cleaned=%.200s", e, cleaned)
        return SkillExtraction(mapped_topics=taxonomy[:3])


async def generate_plan(extraction: SkillExtraction) -> InterviewPlan:
    """Generate interview plan from skill extraction."""
    llm = get_llm(temperature=0.1, max_tokens=1000)
    prompt = PLAN_PROMPT.format(extraction=extraction.model_dump_json())
    response = await llm.ainvoke([
        SystemMessage(content="你是一个精确的 JSON 助手，只输出合法的 JSON，不要输出任何其他内容。"),
        HumanMessage(content=prompt),
    ])

    try:
        data = json.loads(strip_json_fences(response.content))
        return InterviewPlan(**data)
    except (json.JSONDecodeError, Exception) as e:
        logger.error("Failed to parse interview plan: %s", e)
        resume_topics = extraction.mapped_topics[:5]
        random_topics = [t for t in load_topic_taxonomy() if t not in resume_topics][:3]
        return InterviewPlan(
            resume_topics=resume_topics,
            random_topics=random_topics,
            resume_question_count=max(12, len(resume_topics) * 2),
            random_question_count=max(2, 4 - len(resume_topics) // 3),
        )


async def route_resume(state: InterviewState, resume_text: str) -> InterviewPlan:
    """Full router pipeline: extract skills → generate plan → update state."""
    state.resume_text = resume_text

    extraction = await extract_skills(resume_text)
    state.candidate_name = extraction.candidate_name
    state.skill_entities = [
        SkillEntity(name=s.name, source=s.category)
        for s in extraction.skills
    ]

    plan = await generate_plan(extraction)
    state.resume_topics = plan.resume_topics
    state.progress.pending_resume_topics = list(plan.resume_topics)
    state.progress.current_phase = "interviewing"

    logger.info(
        "Router: candidate=%s, resume_topics=%s, random_topics=%s, plan=%d+%d questions",
        state.candidate_name,
        plan.resume_topics,
        plan.random_topics,
        plan.resume_question_count,
        plan.random_question_count,
    )

    return plan
