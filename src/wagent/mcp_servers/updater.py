"""Knowledge Updater Agent – gap analysis → targeted crawl → quality filter → dedup → ingest.

Runs on a schedule (every 3 hours) or on-demand. Uses the MCP server to
search for new interview content and intelligently updates the RAG knowledge base.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from qdrant_client.models import FieldCondition, Filter, MatchValue

from wagent.config import get_settings, load_topic_taxonomy, save_topic_taxonomy
from wagent.llm import get_llm
from wagent.mcp_servers.xiaohongshu_server import XiaohongshuMCPServer
from wagent.rag.chunking import classify_chunk_metadata, semantic_chunk, generate_contextual_description
from wagent.utils import strip_json_fences
from wagent.rag.embeddings import embed_texts
from wagent.rag.store import add_chunks, check_duplicate, collection_stats, get_qdrant_client

logger = logging.getLogger(__name__)

UPDATE_LOG = Path("logs/update_detail.jsonl")

_EMPTY_ANSWERS = {"待补充", "暂无", "略", "无", "N/A", "", "None"}


def _is_valid_answer(answer: str | None) -> bool:
    """Return True only if the answer has real substance."""
    if answer is None:
        return False
    stripped = str(answer).strip()
    if not stripped or len(stripped) < 10 or stripped in _EMPTY_ANSWERS:
        return False
    return True


def _log_ingested_item(item_text: str, meta: dict[str, Any], pipeline: str) -> None:
    """Append a detailed record for each successfully ingested item."""
    UPDATE_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now().isoformat(),
        "source_url": meta.get("original_doc", ""),
        "topic": meta.get("topic", "general"),
        "difficulty": meta.get("difficulty", "basic"),
        "answer_source": meta.get("answer_source", "extracted"),
        "text_preview": item_text[:500],
        "pipeline": pipeline,
    }
    with open(UPDATE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


GAP_ANALYSIS_PROMPT = """\
分析当前知识库的覆盖情况，找出薄弱领域。

当前各 topic 数量分布:
{topic_dist}

Topic 分类体系: {taxonomy}

请找出覆盖最少的 3-5 个 topic，并为它们生成**中文**搜索关键词，
用于在小红书上搜索 AI/大模型面试相关的帖子。

以 JSON 格式回复:
{{
  "weak_topics": ["topic1", "topic2"],
  "search_queries": ["大模型面试 xxx", "AI面试 yyy"]
}}
"""

QA_EXTRACT_PROMPT = """\
从以下内容中提取结构化的面试问答。只提取真正与 AI/大模型/Agent 技术面试相关的内容。

规则：
1. 如果原文同时包含问题和答案，直接提取，answer_source 标记为 "extracted"
2. 如果原文只有问题没有答案，且你确信自己知道准确答案，用你的专业知识给出完整回答，answer_source 标记为 "llm_generated"
3. 如果问题不相关、或你不确定答案的正确性，跳过该问题
4. 答案不得为"待补充"、"暂无"等占位符

内容:
{content}

如果包含有价值的面试问答内容，以 JSON 格式回复:
{{
  "is_relevant": true,
  "qa_pairs": [
    {{"question": "问题内容", "answer": "答案内容", "topic": "所属方向", "answer_source": "extracted|llm_generated"}}
  ]
}}

如果内容不相关（广告、跑题等），回复:
{{"is_relevant": false, "qa_pairs": []}}
"""


TAXONOMY_REVIEW_PROMPT = """\
你是知识库管理助手。以下是当前被标记为 "general"（未归类）的 chunk 样本。
现有方向列表: {taxonomy}

请分析这些 chunk 的内容，判断是否需要新增方向来覆盖它们。

General chunk 样本（最多 20 条）:
{samples}

规则:
- 如果大部分 chunk 确实不属于任何现有方向，提议 1-3 个新方向
- 新方向名称使用**中文**，贴近中国互联网企业 AI 面试的叫法（如"知识图谱"、"对话系统"、"搜索引擎"）
- 如果样本内容其实可以归入现有方向，不要提议新方向
- 只提议在 AI/大模型/面试领域有意义的方向

以 JSON 格式回复:
{{
  "proposed_topics": ["新方向1", "新方向2"],
  "reason": "简要说明为何需要这些新方向",
  "should_add": true
}}

如果不需要新增:
{{
  "proposed_topics": [],
  "reason": "这些 chunk 可以归入现有方向，无需新增",
  "should_add": false
}}
"""

RESUME_GAP_PROMPT = """\
给定候选人简历中提取的技术话题列表和知识库中各话题的现有 chunk 数量，
找出知识库覆盖不足的话题（少于 {threshold} 个 chunk），
并为每个薄弱话题生成**中文**搜索关键词。

候选人相关话题: {resume_topics}
知识库各话题 chunk 数量: {topic_counts}

以 JSON 格式回复:
{{
  "weak_topics": [{{"topic": "topic_name", "current_count": 2, "search_keywords": ["搜索词1", "搜索词2"]}}],
  "summary": "简短总结覆盖情况"
}}
"""


class KnowledgeUpdater:
    """Orchestrates the knowledge base update pipeline."""

    def __init__(self, mcp_server: XiaohongshuMCPServer | None = None):
        self.mcp = mcp_server or XiaohongshuMCPServer()
        self.pending_search_topics: list[str] = []

    async def run_update(self) -> dict[str, Any]:
        """Full update pipeline: gap analysis → search → filter → dedup → ingest."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "searched_queries": 0,
            "raw_results": 0,
            "content_available": 0,
            "content_empty": 0,
            "relevant_extracted": 0,
            "duplicates_skipped": 0,
            "chunks_added": 0,
        }

        # 1. Gap analysis
        logger.info("KnowledgeUpdater: starting gap analysis")
        queries = await self._gap_analysis()

        # 1.5 Random topic exploration — supplement with non-gap topics
        taxonomy = load_topic_taxonomy()
        gap_keywords = {q.split()[-1] for q in queries}
        non_gap = [t for t in taxonomy if t not in gap_keywords and t != "general"]
        explore_count = min(3, len(non_gap))
        if non_gap and explore_count > 0:
            sampled = random.sample(non_gap, explore_count)
            for topic in sampled:
                queries.append(f"AI大模型面试 {topic} 最新面试题")
            logger.info("Random topic exploration: added %s", sampled)

        report["searched_queries"] = len(queries)

        # 2. Search via MCP
        all_results = []
        for query in queries:
            results = await self.mcp.search(query, max_results=10)
            all_results.extend(results)
            await asyncio.sleep(2)
        report["raw_results"] = len(all_results)

        # 3. Quality filter: extract Q&A
        extracted = []
        for result in all_results:
            content = result.get("content", "")
            title = result.get("title", "")
            content_len = len(content)

            if content_len <= len(title) + 5:
                report["content_empty"] += 1
            else:
                report["content_available"] += 1

            qa_data = await self._extract_qa(content)
            if qa_data.get("is_relevant"):
                for qa in qa_data.get("qa_pairs", []):
                    answer = qa.get("answer") or ""
                    if not _is_valid_answer(answer):
                        logger.debug("Skipped QA with empty/short answer: %r", str(answer)[:50])
                        continue
                    extracted.append({
                        "text": f"Q: {qa['question']}\nA: {answer}",
                        "topic": qa.get("topic", "general"),
                        "source_url": result.get("url", ""),
                        "answer_source": qa.get("answer_source", "extracted"),
                    })
        report["relevant_extracted"] = len(extracted)

        logger.info(
            "KnowledgeUpdater: content stats — %d with content, %d empty/title-only out of %d",
            report["content_available"], report["content_empty"], report["raw_results"],
        )

        # 4. Chunk + Contextual Retrieval + Dedup + Ingest
        cfg = get_settings()
        client = get_qdrant_client()
        added = 0
        dupes = 0

        for item in extracted:
            text = item["text"]

            if check_duplicate(client, text, threshold=cfg.dedup_similarity_threshold):
                dupes += 1
                continue

            ctx_desc = await generate_contextual_description(text, text)
            enriched = f"{ctx_desc}\n\n{text}"

            meta = await classify_chunk_metadata(text)
            meta.update({
                "source": "crawled",
                "date_added": datetime.now().isoformat()[:10],
                "original_doc": item.get("source_url", "xiaohongshu"),
                "quality_score": 0.7,
                "answer_source": item.get("answer_source", "extracted"),
            })

            embeddings = embed_texts([enriched])
            add_chunks(client, [enriched], [meta], embeddings)
            _log_ingested_item(text, meta, "run_update")
            added += 1

        report["duplicates_skipped"] = dupes
        report["chunks_added"] = added

        logger.info(
            "KnowledgeUpdater: done. queries=%d, raw=%d, "
            "content=%d/%d, extracted=%d, dupes=%d, added=%d",
            report["searched_queries"],
            report["raw_results"],
            report["content_available"],
            report["raw_results"],
            report["relevant_extracted"],
            report["duplicates_skipped"],
            report["chunks_added"],
        )

        return report

    async def _gap_analysis(self) -> list[str]:
        """Analyze knowledge base to find weak topics and generate search queries."""
        client = get_qdrant_client()
        stats = collection_stats(client)

        taxonomy = load_topic_taxonomy()
        topic_counts: dict[str, int] = {t: 0 for t in taxonomy}

        try:
            points, _ = client.scroll(
                collection_name="knowledge_base", limit=500, with_payload=True
            )
            for point in points:
                topic = point.payload.get("topic", "general")
                if topic in topic_counts:
                    topic_counts[topic] += 1
        except Exception:
            pass

        llm = get_llm(tier="fast", temperature=0.2, max_tokens=500)
        prompt = GAP_ANALYSIS_PROMPT.format(
            topic_dist=json.dumps(topic_counts, ensure_ascii=False),
            taxonomy=json.dumps(taxonomy),
        )
        response = await llm.ainvoke([
            SystemMessage(content="只输出合法的 JSON，不要输出任何其他内容。"),
            HumanMessage(content=prompt),
        ])

        try:
            cleaned = strip_json_fences(response.content)
            data = json.loads(cleaned)
            return data.get("search_queries", [])
        except (json.JSONDecodeError, Exception):
            logger.warning("_gap_analysis: failed to parse LLM response: %s", response.content[:200])
            sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1])
            return [
                f"大模型面试 {t[0]}" for t in sorted_topics[:3]
            ]

    async def _extract_qa(self, content: str) -> dict[str, Any]:
        """Extract structured Q&A from raw crawled content."""
        if not content or len(content) < 20:
            return {"is_relevant": False, "qa_pairs": []}

        llm = get_llm(temperature=0.0, max_tokens=8192)
        prompt = QA_EXTRACT_PROMPT.format(content=content[:50000])
        response = await llm.ainvoke([
            SystemMessage(content="只输出合法的 JSON，不要输出任何其他内容。"),
            HumanMessage(content=prompt),
        ])

        try:
            cleaned = strip_json_fences(response.content)
            return json.loads(cleaned)
        except (json.JSONDecodeError, Exception):
            logger.warning("_extract_qa: failed to parse LLM response: %s", response.content[:200])
            return {"is_relevant": False, "qa_pairs": []}

    async def resume_gap_analysis(
        self,
        resume_topics: list[str],
        *,
        chunk_threshold: int = 5,
    ) -> list[dict[str, Any]]:
        """Analyze knowledge base coverage for resume-specific topics.

        Returns list of weak topics with search keywords:
        [{"topic": str, "current_count": int, "search_keywords": [str]}]
        """
        client = get_qdrant_client()

        topic_counts: dict[str, int] = {}
        try:
            total = collection_stats(client).get("total_points") or 2000
            points, _ = client.scroll(
                collection_name="knowledge_base", limit=total, with_payload=True
            )
            for point in points:
                topic = point.payload.get("topic", "general")
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
        except Exception:
            pass

        resume_counts = {t: topic_counts.get(t, 0) for t in resume_topics}

        llm = get_llm(tier="fast", temperature=0.2, max_tokens=800)
        prompt = RESUME_GAP_PROMPT.format(
            threshold=chunk_threshold,
            resume_topics=json.dumps(resume_topics, ensure_ascii=False),
            topic_counts=json.dumps(resume_counts, ensure_ascii=False),
        )
        response = await llm.ainvoke([
            SystemMessage(content="只输出合法的 JSON，不要输出任何其他内容。"),
            HumanMessage(content=prompt),
        ])

        try:
            cleaned = strip_json_fences(response.content)
            data = json.loads(cleaned)
            return data.get("weak_topics", [])
        except (json.JSONDecodeError, Exception):
            logger.warning("resume_gap_analysis: parse failed: %s", response.content[:200])
            return [
                {
                    "topic": t,
                    "current_count": resume_counts.get(t, 0),
                    "search_keywords": [f"大模型面试 {t}"],
                }
                for t in resume_topics
                if resume_counts.get(t, 0) < chunk_threshold
            ]

    async def fill_gaps(self, search_keywords: list[str]) -> dict[str, Any]:
        """Run targeted search + ingest for specific keywords."""
        report = {
            "searched": len(search_keywords),
            "raw_results": 0,
            "extracted": 0,
            "added": 0,
        }

        all_results = []
        for kw in search_keywords:
            results = await self.mcp.search(kw, max_results=5)
            all_results.extend(results)
            await asyncio.sleep(2)
        report["raw_results"] = len(all_results)

        cfg = get_settings()
        client = get_qdrant_client()

        for result in all_results:
            qa_data = await self._extract_qa(result.get("content", ""))
            if not qa_data.get("is_relevant"):
                continue

            for qa in qa_data.get("qa_pairs", []):
                answer = qa.get("answer") or ""
                if not _is_valid_answer(answer):
                    logger.debug("fill_gaps: skipped QA with empty/short answer: %r", str(answer)[:50])
                    continue

                text = f"Q: {qa['question']}\nA: {answer}"
                report["extracted"] += 1

                if check_duplicate(client, text, threshold=cfg.dedup_similarity_threshold):
                    continue

                ctx_desc = await generate_contextual_description(text, text)
                enriched = f"{ctx_desc}\n\n{text}"
                meta = await classify_chunk_metadata(text)
                meta.update({
                    "source": "crawled",
                    "date_added": datetime.now().isoformat()[:10],
                    "original_doc": result.get("url", "gap_fill"),
                    "quality_score": 0.7,
                    "answer_source": qa.get("answer_source", "extracted"),
                })

                embeddings = embed_texts([enriched])
                add_chunks(client, [enriched], [meta], embeddings)
                _log_ingested_item(text, meta, "fill_gaps")
                report["added"] += 1

        logger.info("fill_gaps: searched=%d, raw=%d, extracted=%d, added=%d",
                     report["searched"], report["raw_results"], report["extracted"], report["added"])
        return report

    # ──────────── Taxonomy Review ────────────

    async def review_taxonomy(
        self,
        *,
        general_threshold: float = 0.15,
        max_samples: int = 20,
        auto_accept: bool = False,
    ) -> dict[str, Any]:
        """Check if 'general' chunks exceed threshold; if so, ask LLM to propose new topics.

        Returns:
            {
                "total_chunks": int,
                "general_count": int,
                "general_ratio": float,
                "proposed_topics": list[str],
                "reason": str,
                "accepted": bool,
            }
        """
        client = get_qdrant_client()
        taxonomy = load_topic_taxonomy()

        total = 0
        general_count = 0
        general_samples: list[str] = []

        try:
            offset = None
            while True:
                points, next_offset = client.scroll(
                    collection_name="knowledge_base",
                    limit=500,
                    offset=offset,
                    with_payload=True,
                )
                for point in points:
                    total += 1
                    if point.payload.get("topic") == "general":
                        general_count += 1
                        if len(general_samples) < max_samples:
                            text = point.payload.get("text", "")
                            general_samples.append(text[:300])
                if next_offset is None:
                    break
                offset = next_offset
        except Exception as e:
            logger.warning("review_taxonomy: scroll failed: %s", e)

        result: dict[str, Any] = {
            "total_chunks": total,
            "general_count": general_count,
            "general_ratio": general_count / total if total else 0.0,
            "proposed_topics": [],
            "reason": "",
            "accepted": False,
        }

        if total == 0 or (general_count / total) <= general_threshold:
            result["reason"] = (
                f"General 占比 {result['general_ratio']:.1%} ≤ 阈值 {general_threshold:.0%}，无需新增 topic"
            )
            logger.info("review_taxonomy: %s", result["reason"])
            return result

        logger.info(
            "review_taxonomy: general=%d/%d (%.1f%%), triggering LLM review",
            general_count, total, result["general_ratio"] * 100,
        )

        llm = get_llm(tier="fast", temperature=0.3, max_tokens=600)
        prompt = TAXONOMY_REVIEW_PROMPT.format(
            taxonomy=json.dumps(taxonomy, ensure_ascii=False),
            samples="\n---\n".join(general_samples),
        )
        response = await llm.ainvoke([
            SystemMessage(content="只输出合法的 JSON，不要输出任何其他内容。"),
            HumanMessage(content=prompt),
        ])

        try:
            cleaned = strip_json_fences(response.content)
            data = json.loads(cleaned)
        except (json.JSONDecodeError, Exception):
            logger.warning("review_taxonomy: failed to parse LLM response: %s", response.content[:200])
            result["reason"] = "LLM 响应解析失败"
            return result

        proposed = data.get("proposed_topics", [])
        should_add = data.get("should_add", False)
        result["proposed_topics"] = proposed
        result["reason"] = data.get("reason", "")

        if not proposed or not should_add:
            logger.info("review_taxonomy: LLM says no new topics needed — %s", result["reason"])
            return result

        existing = set(taxonomy)
        new_topics = [t for t in proposed if t not in existing]

        if not new_topics:
            result["reason"] += " (all proposed topics already exist)"
            logger.info("review_taxonomy: all proposed topics already exist")
            return result

        result["proposed_topics"] = new_topics

        if auto_accept:
            save_topic_taxonomy(taxonomy + new_topics)
            result["accepted"] = True
            logger.info("review_taxonomy: auto-accepted %d new topics: %s", len(new_topics), new_topics)
        else:
            logger.info(
                "review_taxonomy: proposed %d new topics: %s — reason: %s",
                len(new_topics), new_topics, result["reason"],
            )

        return result

    async def reclassify_general_chunks(self, *, batch_size: int = 20) -> dict[str, int]:
        """Re-run classify_chunk_metadata on all 'general' chunks using the latest taxonomy.

        Returns {"total_general": N, "reclassified": M, "still_general": K, "errors": E}
        """
        client = get_qdrant_client()

        general_points: list[tuple[str, str]] = []  # (point_id, text)
        offset = None

        try:
            while True:
                points, next_offset = client.scroll(
                    collection_name="knowledge_base",
                    limit=500,
                    offset=offset,
                    with_payload=True,
                    scroll_filter=Filter(
                        must=[FieldCondition(key="topic", match=MatchValue(value="general"))]
                    ),
                )
                for point in points:
                    text = point.payload.get("text", "")
                    general_points.append((point.id, text))
                if next_offset is None:
                    break
                offset = next_offset
        except Exception as e:
            logger.warning("reclassify_general_chunks: scroll error: %s", e)

        stats = {"total_general": len(general_points), "reclassified": 0, "still_general": 0, "errors": 0}
        logger.info("reclassify_general_chunks: found %d general chunks to reprocess", len(general_points))

        for i in range(0, len(general_points), batch_size):
            batch = general_points[i : i + batch_size]
            tasks = [classify_chunk_metadata(text) for _, text in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for (point_id, _text), result in zip(batch, results):
                if isinstance(result, Exception):
                    logger.warning("reclassify error for point %s: %s", point_id, result)
                    stats["errors"] += 1
                    continue

                new_topic = result.get("topic", "general")
                new_difficulty = result.get("difficulty", "basic")

                if new_topic == "general":
                    stats["still_general"] += 1
                    continue

                try:
                    client.set_payload(
                        collection_name="knowledge_base",
                        payload={"topic": new_topic, "difficulty": new_difficulty},
                        points=[point_id],
                    )
                    stats["reclassified"] += 1
                except Exception as e:
                    logger.warning("reclassify set_payload error for %s: %s", point_id, e)
                    stats["errors"] += 1

            logger.info(
                "reclassify_general_chunks: batch %d/%d done",
                min(i + batch_size, len(general_points)),
                len(general_points),
            )

        logger.info(
            "reclassify_general_chunks: total=%d, reclassified=%d, still_general=%d, errors=%d",
            stats["total_general"], stats["reclassified"], stats["still_general"], stats["errors"],
        )
        return stats
