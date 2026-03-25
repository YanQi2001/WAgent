"""Embedding and reranking models – singleton loaders for bge-large-zh + bge-reranker."""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

_embed_model = None
_reranker_model = None


def get_embedding_model():
    """Load bge-large-zh-v1.5 (cached singleton)."""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer

        from wagent.config import get_settings

        cfg = get_settings()
        logger.info("Loading embedding model: %s", cfg.embedding_model)
        _embed_model = SentenceTransformer(cfg.embedding_model)
    return _embed_model


def get_reranker_model():
    """Load bge-reranker-v2-m3 (cached singleton)."""
    global _reranker_model
    if _reranker_model is None:
        from sentence_transformers import CrossEncoder

        from wagent.config import get_settings

        cfg = get_settings()
        logger.info("Loading reranker model: %s", cfg.reranker_model)
        _reranker_model = CrossEncoder(cfg.reranker_model)
    return _reranker_model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Encode texts to embeddings."""
    model = get_embedding_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def embed_query(query: str) -> np.ndarray:
    """Encode a single query."""
    return embed_texts([query])[0]


def rerank(query: str, documents: list[str], top_k: int = 5) -> list[tuple[int, float]]:
    """Rerank documents for a query. Returns [(original_index, score), ...] sorted by score desc."""
    if not documents:
        return []
    model = get_reranker_model()
    pairs = [(query, doc) for doc in documents]
    scores = model.predict(pairs)
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda x: x[1], reverse=True)
    return indexed[:top_k]
