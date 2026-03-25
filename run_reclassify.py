"""Reclassify chunks with topic='general' using the fixed classify_chunk_metadata."""

import asyncio
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from qdrant_client.models import FieldCondition, Filter, MatchValue

from wagent.rag.chunking import classify_chunk_metadata
from wagent.rag.store import get_qdrant_client, COLLECTION_NAME

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

LLM_CONCURRENCY = 2
LLM_INTER_CALL_DELAY = 1.0
BATCH_SIZE = 30
BATCH_COOLDOWN = 30
SCROLL_BATCH = 100


@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=3, min=10, max=180),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
async def _safe_classify(chunk: str) -> dict:
    await asyncio.sleep(LLM_INTER_CALL_DELAY + random.uniform(0, 0.5))
    return await classify_chunk_metadata(chunk)


async def reclassify_general_chunks():
    client = get_qdrant_client()
    sem = asyncio.Semaphore(LLM_CONCURRENCY)

    general_filter = Filter(must=[
        FieldCondition(key="topic", match=MatchValue(value="general")),
    ])

    all_points = []
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=general_filter,
            limit=SCROLL_BATCH,
            offset=offset,
            with_payload=True,
        )
        all_points.extend(points)
        logger.info("Scrolled %d points so far (batch=%d)", len(all_points), len(points))
        if next_offset is None or len(points) == 0:
            break
        offset = next_offset

    total = len(all_points)
    logger.info("Found %d chunks with topic='general' to reclassify", total)

    if total == 0:
        return

    reclassified = 0
    still_general = 0
    errors = 0

    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch = all_points[batch_start:batch_end]
        logger.info("Processing batch %d-%d / %d", batch_start + 1, batch_end, total)

        async def _process(point):
            async with sem:
                text = point.payload.get("text", "")
                if not text:
                    return point, None
                try:
                    return point, await _safe_classify(text)
                except Exception as e:
                    logger.error("Failed to classify point %s: %s", point.id, e)
                    return point, None

        results = await asyncio.gather(*[_process(p) for p in batch])

        for point, meta in results:
            if meta is None:
                errors += 1
                continue

            new_topic = meta.get("topic", "general")
            new_difficulty = meta.get("difficulty", "basic")

            if new_topic != "general":
                client.set_payload(
                    collection_name=COLLECTION_NAME,
                    payload={"topic": new_topic, "difficulty": new_difficulty},
                    points=[point.id],
                )
                reclassified += 1
            else:
                still_general += 1

        logger.info(
            "Batch done | reclassified=%d, still_general=%d, errors=%d / %d total",
            reclassified, still_general, errors, total,
        )

        if batch_end < total:
            logger.info("Cooldown %ds before next batch...", BATCH_COOLDOWN)
            await asyncio.sleep(BATCH_COOLDOWN)

    logger.info(
        "=== RECLASSIFICATION COMPLETE ===\n"
        "  Total: %d\n"
        "  Reclassified: %d\n"
        "  Still general: %d\n"
        "  Errors: %d",
        total, reclassified, still_general, errors,
    )


if __name__ == "__main__":
    asyncio.run(reclassify_general_chunks())
