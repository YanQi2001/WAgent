"""Unified vector store backed by Qdrant.

Supports both Docker server mode (QDRANT_URL) and local file mode (QDRANT_PATH).
Server mode is preferred because it allows concurrent access from multiple processes.
All knowledge (manual PDFs + crawled content) lives in one collection
with metadata labels for source, topic, difficulty, etc.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from wagent.config import get_settings
from wagent.rag.embeddings import embed_query, embed_texts

logger = logging.getLogger(__name__)

COLLECTION_NAME = "knowledge_base"
VECTOR_DIM = 1024  # bge-large-zh-v1.5 output dimension


_qdrant_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        cfg = get_settings()
        if cfg.qdrant_url:
            _qdrant_client = QdrantClient(url=cfg.qdrant_url, timeout=30)
            logger.info("Connected to Qdrant server at %s", cfg.qdrant_url)
        else:
            path = cfg.qdrant_abs_path
            path.mkdir(parents=True, exist_ok=True)
            _qdrant_client = QdrantClient(path=str(path))
            logger.info("Opened Qdrant local store at %s", path)
    return _qdrant_client


def close_qdrant_client() -> None:
    """Explicitly close the Qdrant client to avoid __del__ shutdown errors."""
    global _qdrant_client
    if _qdrant_client is not None:
        try:
            _qdrant_client.close()
        except Exception:
            pass
        _qdrant_client = None


def ensure_collection(client: QdrantClient) -> None:
    """Create the collection if it doesn't exist."""
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        logger.info("Created collection: %s", COLLECTION_NAME)


def add_chunks(
    client: QdrantClient,
    chunks: list[str],
    metadatas: list[dict[str, Any]],
    embeddings: np.ndarray | None = None,
) -> int:
    """Add chunks to the vector store. Returns number added."""
    ensure_collection(client)

    if embeddings is None:
        embeddings = embed_texts(chunks)

    points = []
    for i, (chunk, meta, emb) in enumerate(zip(chunks, metadatas, embeddings)):
        point_id = str(uuid.uuid4())
        payload = {
            "text": chunk,
            "source": meta.get("source", "manual"),
            "topic": meta.get("topic", "general"),
            "difficulty": meta.get("difficulty", "basic"),
            "date_added": meta.get("date_added", datetime.now().isoformat()[:10]),
            "original_doc": meta.get("original_doc", ""),
            "quality_score": meta.get("quality_score", 1.0),
            "answer_source": meta.get("answer_source", ""),
        }
        points.append(PointStruct(id=point_id, vector=emb.tolist(), payload=payload))

    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        logger.info("Added %d chunks to knowledge base", len(points))

    return len(points)


def search(
    client: QdrantClient,
    query: str,
    *,
    top_k: int = 20,
    source_filter: str | None = None,
    topic_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Dense vector search with optional metadata filters."""
    ensure_collection(client)
    query_vec = embed_query(query).tolist()

    conditions = []
    if source_filter:
        conditions.append(FieldCondition(key="source", match=MatchValue(value=source_filter)))
    if topic_filter:
        conditions.append(FieldCondition(key="topic", match=MatchValue(value=topic_filter)))

    q_filter = Filter(must=conditions) if conditions else None

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=top_k,
        query_filter=q_filter,
    )

    return [
        {
            "text": r.payload.get("text", ""),
            "score": r.score,
            "topic": r.payload.get("topic", ""),
            "source": r.payload.get("source", ""),
            "difficulty": r.payload.get("difficulty", ""),
            "original_doc": r.payload.get("original_doc", ""),
        }
        for r in response.points
    ]


def check_duplicate(
    client: QdrantClient,
    text: str,
    threshold: float = 0.92,
) -> bool:
    """Check if a similar chunk already exists (for dedup)."""
    ensure_collection(client)
    vec = embed_query(text).tolist()
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vec,
        limit=1,
    )
    if response.points and response.points[0].score >= threshold:
        return True
    return False


def count_by_source(client: QdrantClient, source: str) -> int:
    """Count points with a given source metadata value."""
    ensure_collection(client)
    count = 0
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=500,
            offset=offset,
            with_payload=True,
            scroll_filter=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source))]
            ),
        )
        count += len(points)
        if next_offset is None:
            break
        offset = next_offset
    return count


def delete_by_source(client: QdrantClient, source: str) -> int:
    """Delete all points with the given source metadata. Returns count deleted."""
    ensure_collection(client)
    count = count_by_source(client, source)
    if count == 0:
        return 0

    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source))]
        ),
    )
    logger.info("Deleted %d points with source='%s'", count, source)
    return count


def delete_by_date_and_source(client: QdrantClient, source: str, date_added: str) -> int:
    """Delete points matching both source and date_added. Returns count deleted."""
    ensure_collection(client)
    filt = Filter(must=[
        FieldCondition(key="source", match=MatchValue(value=source)),
        FieldCondition(key="date_added", match=MatchValue(value=date_added)),
    ])
    count = 0
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=500,
            offset=offset,
            with_payload=False,
            with_vectors=False,
            scroll_filter=filt,
        )
        count += len(points)
        if next_offset is None:
            break
        offset = next_offset

    if count == 0:
        return 0

    client.delete(collection_name=COLLECTION_NAME, points_selector=filt)
    logger.info("Deleted %d points with source='%s', date_added='%s'", count, source, date_added)
    return count


def collection_stats(client: QdrantClient) -> dict[str, Any]:
    """Get basic stats about the knowledge base."""
    ensure_collection(client)
    info = client.get_collection(COLLECTION_NAME)
    return {
        "total_points": info.points_count,
    }
