"""PDF auto-downloader – detects PDF URLs from web search, downloads, and triggers ingest.

Safety: max 50MB, validates content-type, stores in data/documents/crawled_*.pdf
"""

from __future__ import annotations

import asyncio
import logging
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from wagent.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

MAX_PDF_SIZE = 50 * 1024 * 1024  # 50 MB
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "documents"


def is_pdf_url(url: str) -> bool:
    """Check if a URL likely points to a PDF."""
    return url.lower().rstrip("/").endswith(".pdf")


async def download_pdf(url: str, *, download_dir: Path | None = None) -> Path | None:
    """Download a PDF from URL to local directory.

    Returns the local path on success, None on failure.
    """
    target_dir = download_dir or DOWNLOAD_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"crawled_{timestamp}.pdf"
    target_path = target_dir / filename

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, _sync_download, url, target_path)
        return result
    except Exception as e:
        logger.error("PDF download failed for %s: %s", url, e)
        return None


def _sync_download(url: str, target_path: Path) -> Path | None:
    """Synchronous PDF download with validation."""
    req = urllib.request.Request(url, headers={
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
        ),
    })

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if "pdf" not in content_type.lower() and not url.lower().endswith(".pdf"):
                logger.warning("URL %s has content-type '%s', skipping", url, content_type)
                return None

            content_length = resp.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_PDF_SIZE:
                logger.warning("PDF too large (%s bytes), skipping: %s", content_length, url)
                return None

            data = resp.read(MAX_PDF_SIZE + 1)
            if len(data) > MAX_PDF_SIZE:
                logger.warning("PDF exceeds 50MB, skipping: %s", url)
                return None

            if not data[:5] == b"%PDF-":
                logger.warning("Downloaded file doesn't look like PDF (magic bytes mismatch): %s", url)
                return None

            with open(target_path, "wb") as f:
                f.write(data)

            logger.info("Downloaded PDF (%d bytes): %s → %s", len(data), url, target_path)
            return target_path

    except Exception as e:
        logger.error("Download error for %s: %s", url, e)
        return None


async def download_and_ingest(url: str) -> dict[str, Any]:
    """Download a PDF and run the full ingest pipeline.

    Returns: {"success": bool, "path": str|None, "chunks_added": int}
    """
    result: dict[str, Any] = {"success": False, "path": None, "chunks_added": 0, "url": url}

    path = await download_pdf(url)
    if path is None:
        return result

    result["path"] = str(path)

    try:
        from wagent.rag.ingest import ingest_document
        chunks = await ingest_document(path, source="crawled")
        result["success"] = True
        result["chunks_added"] = chunks
        logger.info("PDF ingested: %s → %d chunks", path.name, chunks)
    except Exception as e:
        logger.error("PDF ingest failed for %s: %s", path, e)

    return result


async def process_search_results_for_pdfs(
    search_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Scan search results for PDF URLs and auto-download + ingest them.

    Returns list of ingest reports for each PDF found.
    """
    reports = []
    for result in search_results:
        url = result.get("url", "")
        is_pdf = result.get("is_pdf", False) or is_pdf_url(url)

        if is_pdf and url.startswith("http"):
            logger.info("PDF detected in search results: %s", url)
            report = await download_and_ingest(url)
            reports.append(report)

    if reports:
        success_count = sum(1 for r in reports if r["success"])
        total_chunks = sum(r["chunks_added"] for r in reports)
        logger.info(
            "PDF auto-ingest: %d/%d PDFs downloaded, %d chunks added",
            success_count, len(reports), total_chunks,
        )

    return reports
