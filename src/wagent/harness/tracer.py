"""Iteration-level tracing – structured JSON observability.

Replaces LangSmith dependency with self-built tracing.
Records every iteration: tool calls, middleware triggers, compaction events, tokens.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ToolCallTrace:
    tool_name: str
    role: str
    allowed: bool
    blocked_reason: str = ""
    latency_ms: float = 0.0
    input_preview: str = ""
    output_preview: str = ""


@dataclass
class MiddlewareTrace:
    name: str
    phase: str  # "pre" or "post"
    triggered: bool = False
    message: str = ""


@dataclass
class IterationTrace:
    iteration: int
    timestamp: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    tool_calls: list[ToolCallTrace] = field(default_factory=list)
    middleware_events: list[MiddlewareTrace] = field(default_factory=list)
    compaction_triggered: bool = False
    question_mode: str = ""
    agent_role: str = ""


class HarnessTracer:
    """Collect and persist iteration-level traces."""

    def __init__(self, session_id: str, log_dir: Path | None = None):
        self.session_id = session_id
        self.traces: list[IterationTrace] = []
        self._log_dir = log_dir or Path("logs")
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._current: IterationTrace | None = None
        self._iter_start: float = 0

    def begin_iteration(self, iteration: int, agent_role: str = "") -> None:
        self._current = IterationTrace(
            iteration=iteration,
            timestamp=datetime.now().isoformat(),
            agent_role=agent_role,
        )
        self._iter_start = time.time()

    def record_tokens(self, input_tokens: int, output_tokens: int) -> None:
        if self._current:
            self._current.input_tokens = input_tokens
            self._current.output_tokens = output_tokens

    def record_tool_call(self, trace: ToolCallTrace) -> None:
        if self._current:
            self._current.tool_calls.append(trace)

    def record_middleware(self, trace: MiddlewareTrace) -> None:
        if self._current:
            self._current.middleware_events.append(trace)

    def record_compaction(self) -> None:
        if self._current:
            self._current.compaction_triggered = True

    def set_question_mode(self, mode: str) -> None:
        if self._current:
            self._current.question_mode = mode

    def end_iteration(self) -> IterationTrace | None:
        if self._current:
            self._current.latency_ms = (time.time() - self._iter_start) * 1000
            self.traces.append(self._current)
            trace = self._current
            self._current = None
            return trace
        return None

    def save(self) -> Path:
        path = self._log_dir / f"trace_{self.session_id}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for t in self.traces:
                f.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")
        logger.info("HarnessTracer: saved %d iterations to %s", len(self.traces), path)
        return path

    def summary(self) -> dict:
        total_input = sum(t.input_tokens for t in self.traces)
        total_output = sum(t.output_tokens for t in self.traces)
        total_latency = sum(t.latency_ms for t in self.traces)
        tool_blocks = sum(
            1 for t in self.traces for tc in t.tool_calls if not tc.allowed
        )
        compactions = sum(1 for t in self.traces if t.compaction_triggered)
        modes = {}
        for t in self.traces:
            modes[t.question_mode] = modes.get(t.question_mode, 0) + 1

        return {
            "session_id": self.session_id,
            "iterations": len(self.traces),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_latency_ms": total_latency,
            "tool_blocks": tool_blocks,
            "compaction_events": compactions,
            "question_mode_distribution": modes,
        }
