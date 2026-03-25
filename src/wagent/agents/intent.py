"""Intent classification agent – routes user input to interview / Q&A / chitchat.

Uses a lightweight LLM call to classify user intent, then asks for confirmation
before entering the selected mode.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from wagent.llm import get_llm
from wagent.utils import strip_json_fences

logger = logging.getLogger(__name__)

INTENT_PROMPT = """\
你是 AI 面试系统的意图分类器。请将用户输入分类为以下三种意图之一：

- "interview"：用户想要开始或继续模拟技术面试。
  触发信号：提到面试、mock interview、模拟面试、考考我、开始面试、帮我准备面试等
- "qa"：用户想要提问一个技术问题并获取解答。
  触发信号：什么是...、怎么理解...、解释一下...、帮我讲讲...、八股、知识点、原理是什么等
- "chitchat"：普通闲聊、打招呼或不相关话题。
  触发信号：你好、hello、在吗、聊聊天等

用户输入: {user_input}

以 JSON 格式回复:
{{"intent": "interview|qa|chitchat", "confidence": 0.95, "reason": "分类理由"}}
"""

INTENT_LABELS = {
    "interview": "面试模式",
    "qa": "知识问答模式",
    "chitchat": "闲聊",
}


async def classify_intent(user_input: str) -> dict[str, str | float]:
    """Classify user intent using LLM.

    Returns: {"intent": str, "confidence": float, "reason": str}
    """
    llm = get_llm(tier="fast", temperature=0.0, max_tokens=200)
    prompt = INTENT_PROMPT.format(user_input=user_input[:500])
    response = await llm.ainvoke([
        SystemMessage(content="你是一个精确的 JSON 分类器，只输出合法的 JSON，不要输出任何其他内容。"),
        HumanMessage(content=prompt),
    ])

    try:
        data = json.loads(strip_json_fences(response.content))
        intent = data.get("intent", "chitchat")
        if intent not in INTENT_LABELS:
            intent = "chitchat"
        return {
            "intent": intent,
            "confidence": float(data.get("confidence", 0.5)),
            "reason": data.get("reason", ""),
        }
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("Intent classification failed: %s", e)
        return {"intent": "chitchat", "confidence": 0.3, "reason": "parse error"}
