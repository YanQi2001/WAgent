"""LLM-driven natural language confirmation for CLI interactions.

Replaces rigid Y/N prompts with free-text input that gets parsed by an LLM,
allowing users to express fine-grained preferences like "只搜索扩散模型和多模态".
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console

from wagent.cli.prompt_utils import prompt_input
from wagent.llm import get_llm
from wagent.utils import strip_json_fences

logger = logging.getLogger(__name__)
console = Console()

SMART_CONFIRM_PROMPT = """\
你是 CLI 交互助手，根据用户的自然语言输入判断他们想执行哪个操作。

## 当前场景
{situation}

## 可选操作
{actions_desc}

## 用户输入
{user_input}

请分析用户意图，选择最匹配的操作，并提取相关参数。
以 JSON 格式回复: {{"action": "操作id", "params": {{提取的参数}}}}

注意：
- 如果用户只说"好"、"行"、"Y"、"是"等肯定词，选择第一个肯定类操作
- 如果用户说"不"、"N"、"跳过"、"不用"等否定词，选择跳过/拒绝类操作
- 如果用户提到了具体的名称/条目，提取到 params 中
- params 中 items 字段用于存放用户指定的具体条目名称列表
"""


async def smart_confirm(
    situation: str,
    actions: list[dict[str, Any]],
    user_input: str,
    available_items: list[str] | None = None,
) -> dict[str, Any]:
    """Parse free-text user input into a structured action using LLM.

    Args:
        situation: Description of the current context (shown to LLM).
        actions: List of possible actions, each with:
            - id: action identifier (e.g. "fill_all")
            - desc: human-readable description
            - has_items: bool, whether this action can accept specific item selection
        user_input: The raw user input string.
        available_items: Optional list of valid item names for fuzzy matching.

    Returns:
        {"action": str, "params": dict} where params may contain "items" list.
    """
    actions_text = "\n".join(
        f"- {a['id']}: {a['desc']}"
        + (f"（可在 params.items 中指定具体条目，可选范围: {available_items}）" if a.get("has_items") and available_items else "")
        for a in actions
    )

    prompt = SMART_CONFIRM_PROMPT.format(
        situation=situation,
        actions_desc=actions_text,
        user_input=user_input,
    )

    try:
        llm = get_llm(tier="fast", temperature=0, max_tokens=300)
        response = await llm.ainvoke([
            SystemMessage(content="你是一个精确的 JSON 分类器，只输出合法的 JSON，不要输出任何其他内容。"),
            HumanMessage(content=prompt),
        ])

        data = json.loads(strip_json_fences(response.content))
        action_id = data.get("action", "")
        params = data.get("params", {})

        valid_ids = {a["id"] for a in actions}
        if action_id not in valid_ids:
            logger.warning("smart_confirm: LLM returned unknown action '%s', falling back", action_id)
            return _fallback(actions, user_input)

        if available_items and "items" in params:
            params["items"] = _fuzzy_match_items(params["items"], available_items)

        return {"action": action_id, "params": params}

    except Exception as e:
        logger.warning("smart_confirm LLM parsing failed: %s", e)
        return _fallback(actions, user_input)


def _fallback(actions: list[dict[str, Any]], user_input: str) -> dict[str, Any]:
    """Rule-based fallback when LLM is unavailable or fails."""
    lower = user_input.strip().lower()

    negative = {"n", "no", "不", "不用", "跳过", "算了", "不了", "不要", "取消"}
    if lower in negative or any(lower.startswith(w) for w in ["不用", "跳过", "算了", "不要", "取消"]):
        reject_actions = [a for a in actions if any(k in a["id"] for k in ("skip", "reject", "exit"))]
        if reject_actions:
            return {"action": reject_actions[0]["id"], "params": {}}

    positive = {"y", "yes", "好", "行", "好的", "是", "是的", "搜", "搜吧", "确认", "可以", "ok"}
    if lower in positive or any(lower.startswith(w) for w in ["好", "行", "是", "搜", "确认", "可以"]):
        accept_actions = [a for a in actions if not any(k in a["id"] for k in ("skip", "reject", "exit"))]
        if accept_actions:
            return {"action": accept_actions[0]["id"], "params": {}}

    return {"action": actions[0]["id"], "params": {}}


def _fuzzy_match_items(raw_items: list[str], available: list[str]) -> list[str]:
    """Fuzzy-match user-specified items against the available list."""
    matched = []
    for raw in raw_items:
        raw_lower = raw.lower().strip()
        for avail in available:
            if raw_lower in avail.lower() or avail.lower() in raw_lower:
                if avail not in matched:
                    matched.append(avail)
                break
    return matched


async def prompt_and_parse(
    prompt_text: str,
    situation: str,
    actions: list[dict[str, Any]],
    available_items: list[str] | None = None,
) -> dict[str, Any]:
    """Show a prompt, collect free-text input, and parse via smart_confirm.

    Convenience wrapper that combines Prompt.ask + smart_confirm.
    """
    user_input = await prompt_input(prompt_text)
    if not user_input.strip():
        reject = [a for a in actions if any(k in a["id"] for k in ("skip", "reject", "exit"))]
        default = reject[0]["id"] if reject else actions[-1]["id"]
        return {"action": default, "params": {}}

    return await smart_confirm(situation, actions, user_input, available_items)
