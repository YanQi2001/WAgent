"""Pydantic schemas for structured output between agents."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SkillExtraction(BaseModel):
    """Structured output from resume skill extraction."""

    candidate_name: str = ""
    skills: list[SkillItem] = Field(default_factory=list)
    project_keywords: list[str] = Field(default_factory=list)
    free_topics: list[str] = Field(default_factory=list)
    mapped_topics: list[str] = Field(default_factory=list)
    unmapped_topics: list[str] = Field(default_factory=list)
    suggested_question_count: int = 15


class SkillItem(BaseModel):
    name: str
    category: str = ""
    proficiency: str = ""


# Fix forward reference
SkillExtraction.model_rebuild()


class InterviewPlan(BaseModel):
    """Plan produced by Router for how to conduct the interview."""

    resume_topics: list[str] = Field(default_factory=list)
    random_topics: list[str] = Field(default_factory=list)
    resume_question_count: int = 10
    random_question_count: int = 5
    total_questions: int = 15


class QuestionRequest(BaseModel):
    """Request from Router to Interviewer: ask about this topic."""

    topic: str
    mode: str = "resume_driven"
    context_from_resume: str = ""
    knowledge_base_context: str = ""
    follow_up_of: str = ""


class AnswerEvaluation(BaseModel):
    """Interviewer's evaluation of a candidate answer."""

    score: float = Field(ge=0, le=10)
    depth: str = "basic"
    should_follow_up: bool = False
    follow_up_direction: str = ""
    notes: str = ""
    is_exit_request: bool = False


class StudyItem(BaseModel):
    """A single weak knowledge point with study suggestions."""

    topic: str
    weakness_summary: str = ""
    suggested_keywords: list[str] = Field(default_factory=list)
    priority: str = "medium"  # high / medium / low


class InterviewScorecard(BaseModel):
    """Final scorecard produced by Judge agent."""

    overall_score: float = Field(ge=0, le=10)
    topic_scores: list[TopicScoreItem] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    recommendation: str = ""
    summary: str = ""
    battle_scars_index: float = Field(default=0.0, ge=0, le=10)
    first_principles_score: float = Field(default=0.0, ge=0, le=10)
    star_completeness: float = Field(default=0.0, ge=0, le=10)
    followup_resilience: float = Field(default=0.0, ge=0, le=10)
    deep_analysis: str = ""
    study_guide: list[StudyItem] = Field(default_factory=list)


class TopicScoreItem(BaseModel):
    topic: str
    score: float = Field(ge=0, le=10)
    notes: str = ""


InterviewScorecard.model_rebuild()
