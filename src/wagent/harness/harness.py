"""Interview Harness – the control plane that wraps around Agent execution.

Architecture:
  Outer loop (Harness): per-turn control – middleware, compaction, budget, tracing
  Inner loop (LangGraph): per-turn reasoning – agent graph traversal, tool calls

The Harness does NOT interfere with LangGraph's internal graph traversal.
It controls what happens *between* conversation turns with the candidate.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Callable, Awaitable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from wagent.config import get_settings
from wagent.harness.budget import BudgetManager
from wagent.harness.context import ContextCompactor
from wagent.harness.middleware import (
    CandidateGuardrail,
    LoopDetectionMiddleware,
    MiddlewarePipeline,
    SelfVerificationGate,
    TopicCoverageMiddleware,
)
from wagent.harness.state import InterviewState, QAPair, QuestionMode
from wagent.harness.tools import ToolRegistry
from wagent.harness.tracer import HarnessTracer, MiddlewareTrace

logger = logging.getLogger(__name__)

# Type for the LangGraph agent callable the harness wraps
# Third arg is an optional context dict (plan, retriever, current_topic, etc.)
AgentCallable = Callable[[InterviewState, list[BaseMessage], dict[str, Any]], Awaitable[dict[str, Any]]]


class InterviewHarness:
    """The control plane that wraps an interview agent.

    Usage:
        harness = InterviewHarness()
        harness.set_agent(my_langgraph_agent_fn)
        response = await harness.turn(state, candidate_message)
    """

    def __init__(
        self,
        *,
        session_id: str | None = None,
        token_budget: int | None = None,
    ):
        cfg = get_settings()
        self.session_id = session_id or str(uuid.uuid4())[:8]

        self.budget = BudgetManager(token_budget=token_budget or cfg.token_budget)
        self.compactor = ContextCompactor(threshold=cfg.compaction_threshold)
        self.tool_registry = ToolRegistry()
        self.tracer = HarnessTracer(session_id=self.session_id)

        self.middleware = MiddlewarePipeline([
            TopicCoverageMiddleware(coverage_threshold=cfg.coverage_suggest_end),
            LoopDetectionMiddleware(),
            CandidateGuardrail(),
            SelfVerificationGate(),
        ])

        self._agent_fn: AgentCallable | None = None
        self._system_prompt: str = ""
        self._agent_context: dict[str, Any] = {}

    def set_agent(self, agent_fn: AgentCallable) -> None:
        """Register the LangGraph agent callable.

        agent_fn signature: async (state, messages, context) -> {"response": str, "input_tokens": int, "output_tokens": int, ...}
        """
        self._agent_fn = agent_fn

    def set_system_prompt(self, prompt: str) -> None:
        self._system_prompt = prompt

    def set_agent_context(self, **kwargs: Any) -> None:
        """Store additional context passed to the agent callable each turn."""
        self._agent_context.update(kwargs)

    async def turn(
        self,
        state: InterviewState,
        candidate_message: str,
    ) -> str:
        """Execute one interview turn through the full harness pipeline.

        Returns the interviewer's response string.
        """
        if self._agent_fn is None:
            raise RuntimeError("No agent registered. Call set_agent() first.")

        iteration = state.iteration_count + 1
        state.iteration_count = iteration
        self.tracer.begin_iteration(iteration, agent_role="interviewer")
        self.tracer.set_question_mode(state.progress.current_mode.value)

        # --- 1. Budget check ---
        if self.budget.is_over_budget(state):
            state.progress.current_phase = "ending"
            self.tracer.end_iteration()
            return (
                "[面试系统] Token 预算已耗尽，面试即将结束。"
                "感谢你的参与，评分报告将随后生成。"
            )

        if self.budget.should_warn(state):
            logger.info("Budget warning: %s", self.budget.budget_status(state))

        # --- 2. Build messages ---
        messages: list[BaseMessage] = []
        if self._system_prompt:
            messages.append(SystemMessage(content=self._system_prompt))

        for m in state.messages:
            if m.get("role") == "human":
                messages.append(HumanMessage(content=m["content"]))
            elif m.get("role") == "ai":
                messages.append(AIMessage(content=m["content"]))
            elif m.get("role") == "system":
                messages.append(SystemMessage(content=m["content"]))

        messages.append(HumanMessage(content=candidate_message))

        # --- 3. Pre-middleware ---
        messages = await self.middleware.run_pre_hooks(state, messages)

        # --- 4. Context compaction ---
        if self.compactor.should_compact(messages):
            messages = self.compactor.compact(state, messages)
            self.tracer.record_compaction()

        # --- 5. Invoke agent (inner loop – LangGraph handles reasoning + tool calls) ---
        result = await self._agent_fn(state, messages, self._agent_context)
        response_text = result.get("response", "")
        input_tokens = result.get("input_tokens", 0)
        output_tokens = result.get("output_tokens", 0)

        # --- 6. Record tokens ---
        self.budget.record_usage(state, input_tokens, output_tokens)
        self.tracer.record_tokens(input_tokens, output_tokens)

        # --- 7. Post-middleware ---
        response_text = await self.middleware.run_post_hooks(state, response_text)

        # --- 8. Update state ---
        state.messages.append({"role": "human", "content": candidate_message})
        state.messages.append({"role": "ai", "content": response_text})

        # --- 9. End iteration ---
        self.tracer.end_iteration()

        return response_text

    async def suggest_end(self, state: InterviewState) -> bool:
        """Check if the harness recommends ending the interview."""
        if self.budget.is_over_budget(state):
            return True
        cfg = get_settings()
        if state.covered_topic_ratio >= cfg.coverage_suggest_end:
            mode_b_count = sum(
                1 for qa in state.qa_history if qa.mode == QuestionMode.RANDOM_BAGU
            )
            min_random = max(2, 20 - len(state.qa_history))
            if mode_b_count >= min_random or state.progress.questions_asked >= 20:
                return True
        return False

    def finalize(self, state: InterviewState) -> dict[str, Any]:
        """Finalize the session: save traces and return summary."""
        trace_path = self.tracer.save()
        summary = self.tracer.summary()
        summary["budget"] = self.budget.budget_status(state)
        summary["trace_file"] = str(trace_path)
        return summary
