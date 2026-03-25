"""Semantic chunking + Contextual Retrieval (Anthropic approach).

Strategy:
  1. Split document into sentences
  2. Compute embeddings for adjacent sentences
  3. Break at points where cosine similarity drops sharply
  4. For each chunk, generate a contextual description via LLM (Contextual Retrieval)
  5. Assign topic and difficulty metadata via LLM
"""

from __future__ import annotations

import json
import logging
import re

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage

from wagent.config import load_topic_taxonomy
from wagent.llm import get_llm
from wagent.rag.embeddings import embed_texts
from wagent.utils import strip_json_fences

logger = logging.getLogger(__name__)


def split_sentences(text: str) -> list[str]:
    """Split Chinese/English text into sentences."""
    splits = re.split(r'(?<=[。！？.!?\n])\s*', text)
    return [s.strip() for s in splits if s.strip()]


def semantic_chunk(
    text: str,
    *,
    percentile_threshold: int = 25,
    min_chunk_size: int = 100,
    max_chunk_size: int = 1500,
) -> list[str]:
    """Split text into semantic chunks based on embedding similarity breakpoints."""
    sentences = split_sentences(text)
    if len(sentences) <= 3:
        return [text] if text.strip() else []

    embeddings = embed_texts(sentences)

    similarities = []
    for i in range(len(embeddings) - 1):
        sim = float(np.dot(embeddings[i], embeddings[i + 1]))
        similarities.append(sim)

    if not similarities:
        return [text]

    threshold = float(np.percentile(similarities, percentile_threshold))

    breakpoints = []
    for i, sim in enumerate(similarities):
        if sim < threshold:
            breakpoints.append(i + 1)

    chunks = []
    start = 0
    for bp in breakpoints:
        chunk_text = " ".join(sentences[start:bp]).strip()
        if len(chunk_text) >= min_chunk_size:
            chunks.append(chunk_text)
        elif chunks:
            chunks[-1] += " " + chunk_text
        else:
            chunks.append(chunk_text)
        start = bp

    remaining = " ".join(sentences[start:]).strip()
    if remaining:
        if chunks and len(remaining) < min_chunk_size:
            chunks[-1] += " " + remaining
        else:
            chunks.append(remaining)

    # Split oversized chunks
    final = []
    for chunk in chunks:
        if len(chunk) > max_chunk_size:
            for i in range(0, len(chunk), max_chunk_size):
                piece = chunk[i : i + max_chunk_size].strip()
                if piece:
                    final.append(piece)
        else:
            final.append(chunk)

    return final


CONTEXT_PROMPT = """\
以下是完整文档的开头部分:
{doc_preview}

以下是该文档中的一个片段:
{chunk}

请用 2-3 句话写一段简短的上下文描述，说明这个片段在整个文档中的位置和角色。
这段描述会被拼接到片段前面，用于提升检索效果。
请使用与片段相同的语言书写。
"""

METADATA_PROMPT = """\
分析以下文本片段，为其分配元数据标签。

可选的方向列表: {topics}

文本片段: {chunk}

以 JSON 格式回复:
{{"topic": "从方向列表中选一个最匹配的", "difficulty": "basic|intermediate|advanced"}}
"""

async def generate_contextual_description(chunk: str, doc_text: str) -> str:
    """Generate a Contextual Retrieval description for a chunk (Anthropic approach)."""
    llm = get_llm(tier="fast", temperature=0.1, max_tokens=200)
    prompt = CONTEXT_PROMPT.format(
        doc_preview=doc_text[:3000],
        chunk=chunk[:1000],
    )
    response = await llm.ainvoke([
        SystemMessage(content="你是一个简洁的文档分析助手。"),
        HumanMessage(content=prompt),
    ])
    return response.content.strip()


async def classify_chunk_metadata(chunk: str) -> dict[str, str]:
    """Assign topic and difficulty metadata to a chunk."""
    topic_list = load_topic_taxonomy()
    llm = get_llm(tier="fast", temperature=0.0, max_tokens=300)
    prompt = METADATA_PROMPT.format(topics=json.dumps(topic_list), chunk=chunk[:500])
    response = await llm.ainvoke([
        SystemMessage(content="只输出合法的 JSON，不要输出任何其他内容。"),
        HumanMessage(content=prompt),
    ])
    try:
        cleaned = strip_json_fences(response.content)
        data = json.loads(cleaned)
        topic = data.get("topic", "general")
        if topic not in topic_list:
            topic = "general"
        return {"topic": topic, "difficulty": data.get("difficulty", "basic")}
    except (json.JSONDecodeError, Exception):
        topic_match = re.search(r'"topic"\s*:\s*"([^"]+)"', response.content)
        diff_match = re.search(r'"difficulty"\s*:\s*"([^"]+)"', response.content)
        topic = topic_match.group(1) if topic_match else "general"
        difficulty = diff_match.group(1) if diff_match else "basic"
        if topic not in topic_list:
            topic = "general"
        if topic_match:
            logger.debug("classify_chunk_metadata: recovered topic=%s via regex", topic)
        else:
            logger.warning("classify_chunk_metadata: failed to parse: %s", response.content[:200])
        return {"topic": topic, "difficulty": difficulty}
