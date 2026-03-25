"""Shared utility functions."""

from __future__ import annotations

import re


def strip_json_fences(text: str) -> str:
    """Extract JSON from LLM output that may contain markdown fences or preamble text.

    Handles cases where the model prepends prose like
    "Here is the JSON:" before the actual ```json ... ``` block.
    Falls back to extracting the outermost {...} or [...] if no fences found.
    """
    if not text:
        return ""
    text = text.strip()

    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    m = re.search(r"(\{.*\})", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    m = re.search(r"(\[.*\])", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    return text
