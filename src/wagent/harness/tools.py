"""Tool registry with per-role allowlists.

Inspired by Vercel's finding: fewer tools = better results.
Each agent role gets a strict allowlist; calls outside it are blocked.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Register tools and enforce per-role allowlists."""

    def __init__(self):
        self._tools: dict[str, Callable] = {}
        self._allowlists: dict[str, set[str]] = {
            "router": {"parse_resume", "extract_skills", "handoff"},
            "interviewer": {"search_knowledge_base", "evaluate_response", "search_bing"},
            "judge": {"read_interview_history", "generate_scorecard"},
            "updater": {"search_xiaohongshu", "search_bing", "ingest_to_rag"},
            "qa_agent": {"search_knowledge_base", "search_bing"},
        }

    def register(self, name: str, fn: Callable) -> None:
        self._tools[name] = fn

    def get_tool(self, name: str) -> Callable | None:
        return self._tools.get(name)

    def is_allowed(self, role: str, tool_name: str) -> bool:
        allowed = self._allowlists.get(role, set())
        return tool_name in allowed

    def call(self, role: str, tool_name: str, **kwargs: Any) -> Any:
        if not self.is_allowed(role, tool_name):
            logger.warning(
                "ToolRegistry: BLOCKED %s calling %s (not in allowlist)", role, tool_name
            )
            return {"error": f"Tool '{tool_name}' not allowed for role '{role}'", "blocked": True}

        fn = self._tools.get(tool_name)
        if fn is None:
            return {"error": f"Tool '{tool_name}' not registered"}
        return fn(**kwargs)

    def get_allowed_tools(self, role: str) -> list[str]:
        return sorted(self._allowlists.get(role, set()))

    def set_allowlist(self, role: str, tools: set[str]) -> None:
        self._allowlists[role] = tools
