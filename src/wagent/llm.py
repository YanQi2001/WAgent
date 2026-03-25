"""Unified LLM client — works with any OpenAI-compatible API provider.

Supported providers (configure via .env):
  DeepSeek, MiniMax, OpenAI, NVIDIA Inference API, Moonshot, etc.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from wagent.config import get_settings


def get_llm(
    *,
    tier: str = "strong",
    temperature: float = 0.3,
    max_tokens: int | None = None,
    streaming: bool = False,
) -> ChatOpenAI:
    """Return a ChatOpenAI instance pointing at the configured LLM endpoint.

    tier="fast"   → cheap/fast model (SiliconFlow DeepSeek V3) for classification & short gen
    tier="strong" → thinking model (NVIDIA Gemini 3 Pro) for complex reasoning & long gen
    Falls back to strong if fast tier is not configured.
    """
    cfg = get_settings()
    if tier == "fast" and cfg.llm_fast_api_key:
        api_key = cfg.llm_fast_api_key
        base_url = cfg.llm_fast_base_url
        model = cfg.llm_fast_model
    else:
        api_key = cfg.llm_api_key
        base_url = cfg.llm_base_url
        model = cfg.llm_model

    kwargs: dict = {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "temperature": temperature,
        "streaming": streaming,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return ChatOpenAI(**kwargs)
