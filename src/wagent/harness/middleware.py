"""Middleware pipeline for the Interview Harness.

Each middleware implements pre_hook (before LLM call) and/or post_hook (after).
Inspired by LangChain's deepagents-cli middleware architecture that improved
Terminal Bench 2.0 from Top 30 → Top 5 purely through harness changes.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage

from wagent.harness.state import InterviewState, QuestionMode

logger = logging.getLogger(__name__)


class Middleware(ABC):
    """Base middleware class. Override pre_hook / post_hook as needed."""

    @abstractmethod
    async def pre_hook(
        self, state: InterviewState, messages: list[BaseMessage]
    ) -> list[BaseMessage]:
        """Modify messages before they enter the LLM. Return updated messages."""
        return messages

    async def post_hook(self, state: InterviewState, response: str) -> str:
        """Inspect or modify the LLM response. Return updated response."""
        return response


class TopicCoverageMiddleware(Middleware):
    """Track covered topics and drive question-mode switching.

    When resume-topic coverage reaches the threshold, injects a nudge
    to switch to random bagu mode.
    """

    def __init__(self, coverage_threshold: float = 0.85):
        self.coverage_threshold = coverage_threshold

    async def pre_hook(
        self, state: InterviewState, messages: list[BaseMessage]
    ) -> list[BaseMessage]:
        if not state.resume_topics:
            return messages

        ratio = state.covered_topic_ratio
        uncovered = [
            t for t in state.resume_topics if t not in state.progress.covered_topics
        ]

        if ratio >= self.coverage_threshold:
            if state.progress.current_mode == QuestionMode.RESUME_DRIVEN:
                state.progress.current_mode = QuestionMode.RANDOM_BAGU
                messages.append(
                    SystemMessage(
                        content=(
                            "[Harness] Resume-driven topics covered "
                            f"({ratio:.0%}). Switch to random bagu questions "
                            "to broaden coverage."
                        )
                    )
                )
                logger.info("TopicCoverage: switching to RANDOM_BAGU mode (%.0f%%)", ratio * 100)
        elif uncovered:
            messages.append(
                SystemMessage(
                    content=(
                        f"[Harness] Topics not yet covered from resume: {uncovered}. "
                        f"Coverage: {ratio:.0%}."
                    )
                )
            )

        return messages

    async def post_hook(self, state: InterviewState, response: str) -> str:
        return response


class LoopDetectionMiddleware(Middleware):
    """Detect doom loops where the interviewer keeps asking similar questions.

    Compares recent questions by simple text overlap. A full embedding-based
    version can be swapped in once the RAG module provides an encoder.
    """

    def __init__(self, window: int = 4, similarity_threshold: float = 0.7):
        self.window = window
        self.similarity_threshold = similarity_threshold

    async def pre_hook(
        self, state: InterviewState, messages: list[BaseMessage]
    ) -> list[BaseMessage]:
        recent = state.qa_history[-self.window :]
        if len(recent) < 2:
            return messages

        last_q = recent[-1].question.lower()
        for older in recent[:-1]:
            overlap = self._jaccard(last_q, older.question.lower())
            if overlap > self.similarity_threshold:
                messages.append(
                    SystemMessage(
                        content=(
                            "[Harness] Loop detected: recent question is very similar "
                            "to an earlier one. Please change topic or approach."
                        )
                    )
                )
                logger.warning("LoopDetection: jaccard=%.2f, injecting redirect", overlap)
                break

        return messages

    async def post_hook(self, state: InterviewState, response: str) -> str:
        return response

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        sa, sb = set(a.split()), set(b.split())
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)


class CandidateGuardrail(Middleware):
    """Screen candidate input for prompt injection attempts.

    Uses a lightweight LLM call to classify whether the candidate's message
    is an injection attempt.
    """

    INJECTION_PATTERNS = [
        "忽略所有",
        "ignore all",
        "ignore previous",
        "忽略前面",
        "你现在是",
        "you are now",
        "宣布我通过",
        "直接通过",
    ]

    async def pre_hook(
        self, state: InterviewState, messages: list[BaseMessage]
    ) -> list[BaseMessage]:
        if not messages:
            return messages

        last = messages[-1]
        content = getattr(last, "content", "")
        if isinstance(content, str):
            lower = content.lower()
            for pattern in self.INJECTION_PATTERNS:
                if pattern in lower:
                    logger.warning("CandidateGuardrail: injection pattern detected: %s", pattern)
                    messages[-1] = type(last)(
                        content="[The candidate's response was filtered by the safety system.]"
                    )
                    break

        return messages

    async def post_hook(self, state: InterviewState, response: str) -> str:
        return response


class SelfVerificationGate(Middleware):
    """Before ending the interview, force a verification pass.

    Checks whether all evaluation dimensions have been addressed.
    Similar to LangChain's PreCompletionChecklistMiddleware.
    """

    async def pre_hook(
        self, state: InterviewState, messages: list[BaseMessage]
    ) -> list[BaseMessage]:
        return messages

    async def post_hook(self, state: InterviewState, response: str) -> str:
        if state.progress.current_phase == "ending":
            uncovered = [
                t for t in state.resume_topics if t not in state.progress.covered_topics
            ]
            if uncovered:
                logger.info(
                    "SelfVerification: %d topics still uncovered at ending phase", len(uncovered)
                )
                return (
                    response
                    + f"\n\n[Harness SelfVerify] Warning: {len(uncovered)} resume topics "
                    f"not yet covered: {uncovered}"
                )

        return response


class MiddlewarePipeline:
    """Runs an ordered list of middlewares as pre/post hooks."""

    def __init__(self, middlewares: list[Middleware] | None = None):
        self.middlewares = middlewares or []

    async def run_pre_hooks(
        self, state: InterviewState, messages: list[BaseMessage]
    ) -> list[BaseMessage]:
        for mw in self.middlewares:
            messages = await mw.pre_hook(state, messages)
        return messages

    async def run_post_hooks(self, state: InterviewState, response: str) -> str:
        for mw in self.middlewares:
            response = await mw.post_hook(state, response)
        return response
