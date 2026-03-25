"""Document ingestion pipeline: PDF → Semantic Chunking → Contextual Retrieval → Qdrant."""

from __future__ import annotations

import asyncio
import logging
import random
import warnings
from datetime import datetime
from pathlib import Path

import pdfplumber
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from wagent.rag.chunking import (
    classify_chunk_metadata,
    generate_contextual_description,
    semantic_chunk,
)
from wagent.rag.embeddings import embed_texts
from wagent.rag.store import add_chunks, check_duplicate, get_qdrant_client

logger = logging.getLogger(__name__)

LLM_CONCURRENCY = 2
LLM_INTER_CALL_DELAY = 1.0
BATCH_SIZE = 50
BATCH_COOLDOWN = 30


def extract_pdf_text(path: str | Path) -> str:
    """Extract text from PDF."""
    text_parts = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*FontBBox.*")
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
    return "\n".join(text_parts)


async def ingest_document(
    path: str | Path,
    *,
    source: str = "manual",
    enable_contextual: bool = True,
    start_chunk_offset: int = 0,
) -> int:
    """Ingest a single document into the knowledge base.

    Args:
        start_chunk_offset: Skip the first N chunks (for resuming after crash).

    Returns number of chunks added.
    """
    path = Path(path)
    logger.info("Ingesting %s (source=%s)", path.name, source)

    # 1. Extract text
    if path.suffix.lower() == ".pdf":
        doc_text = extract_pdf_text(path)
    else:
        doc_text = path.read_text(encoding="utf-8")

    if not doc_text.strip():
        logger.warning("Empty document: %s", path)
        return 0

    # 2. Semantic chunking
    chunks = semantic_chunk(doc_text)
    logger.info("Chunked into %d pieces", len(chunks))

    # 3. Contextual Retrieval + metadata classification (batched with rate limiting)
    total_chunks = len(chunks)
    sem = asyncio.Semaphore(LLM_CONCURRENCY)

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=3, min=10, max=180),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _safe_contextual(chunk: str, doc: str) -> str:
        await asyncio.sleep(LLM_INTER_CALL_DELAY + random.uniform(0, 0.5))
        return await generate_contextual_description(chunk, doc)

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=3, min=10, max=180),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _safe_classify(chunk: str) -> dict:
        await asyncio.sleep(LLM_INTER_CALL_DELAY + random.uniform(0, 0.5))
        return await classify_chunk_metadata(chunk)

    async def _process_chunk(idx: int, chunk: str) -> tuple[str, dict]:
        async with sem:
            if enable_contextual:
                ctx_desc = await _safe_contextual(chunk, doc_text)
                enriched = f"{ctx_desc}\n\n{chunk}"
            else:
                enriched = chunk

            meta = await _safe_classify(chunk)
            meta.update({
                "source": source,
                "date_added": datetime.now().isoformat()[:10],
                "original_doc": path.name,
                "quality_score": 1.0 if source == "manual" else 0.8,
            })
            return enriched, meta

    client = get_qdrant_client()
    from wagent.config import get_settings
    cfg = get_settings()
    total_added = 0
    total_dupes = 0

    if start_chunk_offset > 0:
        logger.info("Resuming from chunk %d (skipping first %d)", start_chunk_offset, start_chunk_offset)

    for batch_start in range(start_chunk_offset, total_chunks, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_chunks)
        batch = chunks[batch_start:batch_end]
        logger.info(
            "Processing batch %d-%d / %d [%s]",
            batch_start + 1, batch_end, total_chunks, path.name,
        )

        results = await asyncio.gather(
            *[_process_chunk(batch_start + i, c) for i, c in enumerate(batch)]
        )

        enriched_batch = [r[0] for r in results]
        meta_batch = [r[1] for r in results]

        final_chunks = []
        final_metas = []
        for chunk, meta in zip(enriched_batch, meta_batch):
            if check_duplicate(client, chunk, threshold=cfg.dedup_similarity_threshold):
                total_dupes += 1
                continue
            final_chunks.append(chunk)
            final_metas.append(meta)

        if final_chunks:
            embeddings = embed_texts(final_chunks)
            added = add_chunks(client, final_chunks, final_metas, embeddings)
            total_added += added

        logger.info(
            "Batch done: +%d chunks (dupes=%d) | Total so far: %d [%s]",
            len(final_chunks), total_dupes, total_added, path.name,
        )

        if batch_end < total_chunks:
            logger.info("Cooldown %ds before next batch...", BATCH_COOLDOWN)
            await asyncio.sleep(BATCH_COOLDOWN)

    if total_dupes:
        logger.info("Dedup total: skipped %d duplicate chunks", total_dupes)

    logger.info("Added %d chunks from %s", total_added, path.name)
    return total_added


async def run_ingest(directory: str, source: str = "manual") -> None:
    """Ingest all PDFs from a directory."""
    from rich.console import Console
    from qdrant_client.models import FieldCondition, Filter, MatchValue
    from wagent.rag.store import ensure_collection, collection_stats, COLLECTION_NAME

    console = Console()
    dir_path = Path(directory)
    if not dir_path.exists():
        console.print(f"[red]Directory not found: {directory}[/red]")
        return

    doc_files = list(dir_path.glob("*.pdf")) + list(dir_path.glob("*.txt"))
    if not doc_files:
        console.print(f"[yellow]No PDF/TXT files found in {directory}[/yellow]")
        return

    client = get_qdrant_client()
    ensure_collection(client)

    total = 0
    skipped = 0
    for pdf in doc_files:
        existing = client.count(
            collection_name=COLLECTION_NAME,
            count_filter=Filter(must=[
                FieldCondition(key="original_doc", match=MatchValue(value=pdf.name))
            ]),
        ).count
        if existing > 0:
            console.print(f"[dim]跳过 {pdf.name}（已入库 {existing} chunks）[/dim]")
            skipped += 1
            continue

        console.print(f"[dim]Processing {pdf.name}...[/dim]")
        count = await ingest_document(pdf, source=source)
        total += count
        console.print(f"  [green]+{count} chunks[/green]")

    stats = collection_stats(client)
    console.print(f"\n[bold green]Ingestion complete: {total} chunks added, {skipped} files skipped[/bold green]")
    console.print(f"[dim]Knowledge base total: {stats['total_points']} chunks[/dim]")
