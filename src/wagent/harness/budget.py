"""Budget manager – tracks token usage and cost per interview session."""

from __future__ import annotations

import logging

from wagent.harness.state import InterviewState

logger = logging.getLogger(__name__)


class BudgetManager:
    """Enforces token and cost budgets for an interview session."""

    def __init__(self, token_budget: int = 100_000):
        self.token_budget = token_budget

    def record_usage(
        self,
        state: InterviewState,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        state.total_input_tokens += input_tokens
        state.total_output_tokens += output_tokens

    @property
    def _warning_threshold(self) -> float:
        return 0.85

    def is_over_budget(self, state: InterviewState) -> bool:
        total = state.total_input_tokens + state.total_output_tokens
        return total >= self.token_budget

    def should_warn(self, state: InterviewState) -> bool:
        total = state.total_input_tokens + state.total_output_tokens
        return total >= int(self.token_budget * self._warning_threshold)

    def budget_status(self, state: InterviewState) -> str:
        total = state.total_input_tokens + state.total_output_tokens
        pct = total / self.token_budget * 100 if self.token_budget else 0
        return (
            f"Tokens: {total:,}/{self.token_budget:,} ({pct:.1f}%) | "
            f"In: {state.total_input_tokens:,} Out: {state.total_output_tokens:,} | "
            f"Iterations: {state.iteration_count}"
        )
