#!/usr/bin/env python
"""One-off script to ingest documents with visible progress.

Usage:
    python run_ingest.py                  # Ingest all from scratch
    python run_ingest.py --resume 550     # Resume big PDF from chunk 550
"""
import argparse
import asyncio
import logging
import sys

sys.path.insert(0, "src")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    stream=sys.stderr,
)

from pathlib import Path
from wagent.rag.ingest import ingest_document, run_ingest  # noqa: E402
from wagent.rag.store import get_qdrant_client, collection_stats  # noqa: E402


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=int, default=0,
                        help="Resume big PDF from chunk offset N")
    args = parser.parse_args()

    doc_dir = Path("data/documents")
    docs = sorted(doc_dir.glob("*.pdf")) + sorted(doc_dir.glob("*.txt"))

    total = 0
    for doc in docs:
        offset = args.resume if doc.stat().st_size > 5_000_000 and args.resume > 0 else 0
        logging.info("=== Processing %s (offset=%d) ===", doc.name, offset)
        count = await ingest_document(doc, source="manual", start_chunk_offset=offset)
        total += count
        logging.info("=== %s done: +%d chunks ===", doc.name, count)

    client = get_qdrant_client()
    stats = collection_stats(client)
    logging.info("TOTAL: %d new chunks added. KB total: %d", total, stats["total_points"])


asyncio.run(main())
