"""One-time migration: rename English topic names to Chinese in Qdrant."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from qdrant_client.models import FieldCondition, Filter, MatchValue

from wagent.rag.store import get_qdrant_client, COLLECTION_NAME

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

TOPIC_MIGRATION = {
    "transformer_architecture": "Transformer架构",
    "llm_training": "大模型训练",
    "llm_inference": "大模型推理与部署",
    "rag_pipeline": "RAG检索增强生成",
    "prompt_engineering": "提示工程",
    "agent_architecture": "Agent智能体",
    "vector_database": "向量数据库",
    "fine_tuning": "微调技术",
    "evaluation_methods": "模型评估",
    "deployment": "大模型推理与部署",
}


def migrate():
    client = get_qdrant_client()
    total_migrated = 0

    for old_topic, new_topic in TOPIC_MIGRATION.items():
        offset = None
        point_ids = []

        while True:
            points, next_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=500,
                offset=offset,
                with_payload=False,
                scroll_filter=Filter(
                    must=[FieldCondition(key="topic", match=MatchValue(value=old_topic))]
                ),
            )
            point_ids.extend([p.id for p in points])
            if next_offset is None:
                break
            offset = next_offset

        if not point_ids:
            logger.info("  %s → %s: 0 points (skip)", old_topic, new_topic)
            continue

        client.set_payload(
            collection_name=COLLECTION_NAME,
            payload={"topic": new_topic},
            points=point_ids,
        )
        total_migrated += len(point_ids)
        logger.info("  %s → %s: %d points migrated", old_topic, new_topic, len(point_ids))

    logger.info("=== Migration complete: %d points updated ===", total_migrated)


if __name__ == "__main__":
    migrate()
