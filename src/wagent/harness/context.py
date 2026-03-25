"""Context compaction engine – three-tier strategy inspired by Manus.

Tiers:
  1. Raw Context (preferred) – recent N turns as-is
  2. Compacted – older turns stripped to key Q&A, retaining restore paths
  3. Summarized (last resort) – earliest turns condensed to structured digest
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from wagent.harness.state import InterviewState

logger = logging.getLogger(__name__)

RAW_WINDOW = 6  # keep last N turns as raw
COMPACT_WINDOW = 10  # next M turns in compacted form


class ContextCompactor:
    """Manages the three-tier context window."""

    def __init__(self, *, model_context_limit: int = 128_000, threshold: float = 0.70):
        self.model_context_limit = model_context_limit
        self.threshold = threshold
        self._summary_cache: str = ""

    def estimate_tokens(self, messages: list[BaseMessage]) -> int:
        """Rough token estimation: 1 token ≈ 1.5 Chinese characters or 4 English chars."""
        total = 0
        for m in messages:
            content = m.content if isinstance(m.content, str) else str(m.content)
            total += max(len(content) // 2, len(content.encode("utf-8")) // 3)
        return total

    def should_compact(self, messages: list[BaseMessage]) -> bool:
        est = self.estimate_tokens(messages)
        return est > int(self.model_context_limit * self.threshold)

    def compact(
        self,
        state: InterviewState,
        messages: list[BaseMessage],
        llm_summarize=None,
    ) -> list[BaseMessage]:
        """Apply three-tier compaction to the message list.

        Args:
            state: current interview state (for progress file injection)
            messages: full conversation messages
            llm_summarize: optional async callable(str) -> str for summarization
        """
        if not self.should_compact(messages):
            return self._inject_progress(state, messages)

        logger.info(
            "ContextCompactor: triggering compaction (est %d tokens, limit %d)",
            self.estimate_tokens(messages),
            self.model_context_limit,
        )

        sys_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        conv_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

        if len(conv_msgs) <= RAW_WINDOW:
            return self._inject_progress(state, messages)

        raw_part = conv_msgs[-RAW_WINDOW:]

        older = conv_msgs[:-RAW_WINDOW]
        compact_part = older[-COMPACT_WINDOW:] if len(older) > COMPACT_WINDOW else older
        summary_part = older[:-COMPACT_WINDOW] if len(older) > COMPACT_WINDOW else []

        compacted_msgs: list[BaseMessage] = []
        compacted_msgs.extend(sys_msgs)

        if summary_part:
            digest = self._build_summary(summary_part)
            compacted_msgs.append(
                SystemMessage(content=f"[Harness 早期对话摘要]\n{digest}")
            )

        for m in compact_part:
            compacted_msgs.append(self._strip_message(m))

        compacted_msgs.extend(raw_part)

        logger.info(
            "ContextCompactor: %d → %d messages (saved ~%d tokens)",
            len(messages),
            len(compacted_msgs),
            self.estimate_tokens(messages) - self.estimate_tokens(compacted_msgs),
        )

        return self._inject_progress(state, compacted_msgs)

    def _inject_progress(
        self, state: InterviewState, messages: list[BaseMessage]
    ) -> list[BaseMessage]:
        """Inject progress file at context start to combat Lost-in-the-Middle."""
        progress_text = (
            f"[面试进度]\n{state.progress.summary()}\n"
            f"累计问答: {len(state.qa_history)} 轮"
        )
        progress_msg = SystemMessage(content=progress_text)

        has_system = any(isinstance(m, SystemMessage) for m in messages)
        if has_system:
            result = []
            inserted = False
            for m in messages:
                result.append(m)
                if isinstance(m, SystemMessage) and not inserted:
                    result.append(progress_msg)
                    inserted = True
            return result
        return [progress_msg] + messages

    def _strip_message(self, msg: BaseMessage) -> BaseMessage:
        """Compact a message: keep first 200 chars + ellipsis."""
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if len(content) > 300:
            content = content[:200] + "...[compacted]"
        return type(msg)(content=content)

    def _build_summary(self, msgs: list[BaseMessage]) -> str:
        """Build a quick structural summary of early messages."""
        lines = []
        for i, m in enumerate(msgs):
            role = "Q" if isinstance(m, HumanMessage) else "A"
            content = m.content if isinstance(m.content, str) else str(m.content)
            preview = content[:80].replace("\n", " ")
            lines.append(f"  {role}{i + 1}: {preview}...")
        return "\n".join(lines)
