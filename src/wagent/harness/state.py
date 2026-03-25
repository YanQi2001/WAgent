"""Shared state schema used across the entire interview session."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class QuestionMode(str, Enum):
    RESUME_DRIVEN = "resume_driven"
    RANDOM_BAGU = "random_bagu"


class SkillEntity(BaseModel):
    name: str
    source: str = ""
    years: float | None = None


class TopicScore(BaseModel):
    topic: str
    score: float = 0.0
    notes: str = ""


class QAPair(BaseModel):
    question: str
    answer: str
    topic: str = ""
    mode: QuestionMode = QuestionMode.RESUME_DRIVEN
    follow_up_depth: int = 0
    score: float | None = None


class ProgressFile(BaseModel):
    """Persistent progress file injected at context start (Manus pattern)."""

    current_phase: str = "init"
    covered_topics: list[str] = Field(default_factory=list)
    pending_resume_topics: list[str] = Field(default_factory=list)
    candidate_strengths: list[str] = Field(default_factory=list)
    candidate_weaknesses: list[str] = Field(default_factory=list)
    questions_asked: int = 0
    current_mode: QuestionMode = QuestionMode.RESUME_DRIVEN

    def summary(self) -> str:
        return (
            f"Phase: {self.current_phase} | "
            f"Questions: {self.questions_asked} | "
            f"Mode: {self.current_mode.value} | "
            f"Covered: {self.covered_topics} | "
            f"Pending: {self.pending_resume_topics}"
        )


class InterviewState(BaseModel):
    """Full interview session state."""

    session_id: str = ""
    started_at: datetime = Field(default_factory=datetime.now)

    # Candidate info
    candidate_name: str = ""
    resume_text: str = ""
    skill_entities: list[SkillEntity] = Field(default_factory=list)
    resume_topics: list[str] = Field(default_factory=list)

    # Interview progress
    progress: ProgressFile = Field(default_factory=ProgressFile)
    qa_history: list[QAPair] = Field(default_factory=list)
    topic_scores: list[TopicScore] = Field(default_factory=list)

    # Harness metadata
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    iteration_count: int = 0

    # Conversation messages (raw LangChain messages)
    messages: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def covered_topic_ratio(self) -> float:
        if not self.resume_topics:
            return 1.0
        covered = set(self.progress.covered_topics) & set(self.resume_topics)
        return len(covered) / len(self.resume_topics)
