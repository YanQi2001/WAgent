"""Hybrid retrieval pipeline: BM25 + Dense + RRF Fusion + Cross-Encoder Re-ranking."""

from __future__ import annotations

import logging
from typing import Any

from rank_bm25 import BM25Okapi

from wagent.rag.embeddings import rerank
from wagent.rag.store import get_qdrant_client, search as dense_search

logger = logging.getLogger(__name__)


def _tokenize_chinese(text: str) -> list[str]:
    """Simple character-level + whitespace tokenization for Chinese BM25."""
    tokens = []
    for word in text.split():
        if any("\u4e00" <= c <= "\u9fff" for c in word):
            tokens.extend(list(word))
        else:
            tokens.append(word.lower())
    return tokens


class HybridRetriever:
    """Three-stage retrieval: BM25 + Dense → RRF Fusion → Cross-Encoder Re-rank."""

    def __init__(
        self,
        *,
        bm25_k: int = 20,
        dense_k: int = 20,
        fusion_k: int = 60,
        rerank_top_k: int = 5,
    ):
        self.bm25_k = bm25_k
        self.dense_k = dense_k
        self.fusion_k = fusion_k
        self.rerank_top_k = rerank_top_k
        self._bm25_corpus: list[dict[str, Any]] = []
        self._bm25_index: BM25Okapi | None = None

    def build_bm25_index(self, documents: list[dict[str, Any]]) -> None:
        """Build in-memory BM25 index from document dicts with 'text' field."""
        self._bm25_corpus = documents
        tokenized = [_tokenize_chinese(doc["text"]) for doc in documents]
        self._bm25_index = BM25Okapi(tokenized)
        logger.info("BM25 index built with %d documents", len(documents))

    def bm25_search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """BM25 sparse retrieval."""
        if self._bm25_index is None:
            return []
        k = top_k or self.bm25_k
        tokens = _tokenize_chinese(query)
        scores = self._bm25_index.get_scores(tokens)

        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        results = []
        for idx, score in indexed:
            if score > 0:
                doc = dict(self._bm25_corpus[idx])
                doc["bm25_score"] = float(score)
                results.append(doc)
        return results

    def retrieve(
        self,
        query: str,
        *,
        source_filter: str | None = None,
        topic_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Full hybrid retrieval pipeline."""
        # Stage 1a: Dense search (Qdrant)
        client = get_qdrant_client()
        dense_results = dense_search(
            client,
            query,
            top_k=self.dense_k,
            source_filter=source_filter,
            topic_filter=topic_filter,
        )

        # Stage 1b: BM25 search
        bm25_results = self.bm25_search(query, top_k=self.bm25_k)

        # Stage 2: RRF Fusion
        fused = self._rrf_fusion(dense_results, bm25_results)

        # Stage 3: Cross-Encoder Re-ranking
        if len(fused) <= 1:
            return fused

        texts = [doc["text"] for doc in fused]
        reranked = rerank(query, texts, top_k=self.rerank_top_k)

        final = []
        for orig_idx, score in reranked:
            doc = dict(fused[orig_idx])
            doc["rerank_score"] = float(score)
            final.append(doc)

        logger.info(
            "HybridRetriever: dense=%d, bm25=%d, fused=%d, reranked=%d",
            len(dense_results),
            len(bm25_results),
            len(fused),
            len(final),
        )

        return final

    def _rrf_fusion(
        self,
        dense: list[dict[str, Any]],
        sparse: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Reciprocal Rank Fusion."""
        k = self.fusion_k
        scores: dict[str, float] = {}
        doc_map: dict[str, dict[str, Any]] = {}

        for rank, doc in enumerate(dense):
            key = doc["text"][:100]
            scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
            doc_map[key] = doc

        for rank, doc in enumerate(sparse):
            key = doc["text"][:100]
            scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
            if key not in doc_map:
                doc_map[key] = doc

        sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        fused = []
        for key in sorted_keys[: self.dense_k + self.bm25_k]:
            doc = dict(doc_map[key])
            doc["rrf_score"] = scores[key]
            fused.append(doc)

        return fused
