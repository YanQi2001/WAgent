"""Terminal input with full CJK support via prompt_toolkit.

Replaces Rich Prompt.ask() which has known issues with Chinese character
backspace, cursor movement, and deletion (Rich #2293, #3374).
"""

from __future__ import annotations

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory

_sessions: dict[str, PromptSession] = {}


def get_prompt_session(name: str = "default") -> PromptSession:
    """Return a named PromptSession singleton (with per-session input history)."""
    if name not in _sessions:
        _sessions[name] = PromptSession(history=InMemoryHistory())
    return _sessions[name]


async def prompt_input(label: str, session_name: str = "default") -> str:
    """Prompt for user input with colored label. Drop-in replacement for Prompt.ask.

    Uses prompt_async() to cooperate with the already-running asyncio event loop
    (all CLI entry points use asyncio.run()).
    """
    session = get_prompt_session(session_name)
    return await session.prompt_async(HTML(f"\n<ansigreen><b>{label}</b></ansigreen>: "))
