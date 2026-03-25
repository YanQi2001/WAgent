"""MCP Server for Xiaohongshu (小红书) search.

Plan A: xiaohongshu-skills CLI (CDP-based, community-maintained).
Plan B fallback: Generic web search via requests.

Exposes a standardized MCP tool interface so the upper-layer Agent is
agnostic to the underlying implementation.

Image OCR: Uses RapidOCR (ONNX-based, local CPU inference) to extract
text from post images, which often contain the bulk of interview Q&A content.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from wagent.config import PROJECT_ROOT

logger = logging.getLogger(__name__)


def _get_ocr_engine():
    """Lazy-init global singleton RapidOCR engine."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.getLogger("RapidOCR").setLevel(logging.WARNING)
        from rapidocr import RapidOCR
        return RapidOCR()


_ocr_engine = None

SKILLS_DIR = PROJECT_ROOT / "tools" / "xiaohongshu-skills"


@dataclass
class SearchResult:
    title: str
    content: str
    url: str
    likes: int = 0


class XiaohongshuMCPServer:
    """MCP-style server wrapping Xiaohongshu search functionality."""

    def __init__(self, skills_dir: Path | None = None):
        self._skills_dir = skills_dir or SKILLS_DIR

    async def search(self, query: str, max_results: int = 10) -> list[dict[str, Any]]:
        """MCP tool: search_xiaohongshu."""
        try:
            results = await self._xhs_skills_search(query, max_results)
        except Exception as e:
            logger.warning("Plan A (xhs-skills) failed: %s. Falling back to Plan B.", e)
            results = await self._fallback_search(query, max_results)

        return [
            {"title": r.title, "content": r.content, "url": r.url, "likes": r.likes}
            for r in results
        ]

    async def _xhs_skills_search(
        self, query: str, max_results: int
    ) -> list[SearchResult]:
        """Plan A: search via CLI, then batch-fetch details in a single session."""
        cli_path = self._skills_dir / "scripts" / "cli.py"
        if not cli_path.exists():
            raise FileNotFoundError(f"xiaohongshu-skills CLI not found at {cli_path}")

        cmd = [
            sys.executable,
            str(cli_path),
            "search-feeds",
            "--keyword", query,
            "--sort-by", "最多点赞",
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._skills_dir / "scripts"),
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)

        if proc.returncode != 0:
            err_msg = stderr.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(f"xhs-skills search failed (exit={proc.returncode}): {err_msg}")

        data = json.loads(stdout.decode("utf-8"))
        feeds = data.get("feeds", [])[:max_results]

        feed_entries = [
            {"feedId": f.get("id", ""), "xsecToken": f.get("xsecToken", "")}
            for f in feeds
            if f.get("id") and f.get("xsecToken")
        ]

        detail_map: dict[str, dict] = {}
        detail_errors: list[str] = []
        if feed_entries:
            detail_map, detail_errors = await self._batch_fetch_details(
                cli_path, feed_entries
            )

        results: list[SearchResult] = []
        ocr_fallback_count = 0
        for feed in feeds:
            title = feed.get("displayTitle", "")
            feed_id = feed.get("id", "")
            likes = self._extract_likes(feed)
            url = f"https://www.xiaohongshu.com/explore/{feed_id}" if feed_id else ""

            content = ""
            if feed_id in detail_map:
                content = await self._extract_content_from_detail(detail_map[feed_id])

            if not content:
                search_images = feed.get("imageList", [])
                if search_images:
                    image_urls = [img.get("url", "") for img in search_images if img.get("url")]
                    if image_urls:
                        ocr_text = await self._ocr_images(image_urls)
                        if ocr_text:
                            content = f"{title}\n\n[图片内容]\n{ocr_text}" if title else ocr_text
                            ocr_fallback_count += 1

            if not content:
                cover_url = feed.get("cover", "")
                if cover_url and not content:
                    ocr_text = await self._ocr_images([cover_url])
                    if ocr_text:
                        content = f"{title}\n\n[封面内容]\n{ocr_text}" if title else ocr_text
                        ocr_fallback_count += 1

            if not content:
                content = title

            if title or content:
                results.append(SearchResult(
                    title=title,
                    content=content,
                    url=url,
                    likes=likes,
                ))

        success = len(detail_map)
        fail = len(detail_errors)
        logger.info(
            "xhs-skills search: %d results for '%s' (detail: %d ok, %d failed, %d ocr-fallback)",
            len(results), query, success, fail, ocr_fallback_count,
        )
        return results

    async def _batch_fetch_details(
        self, cli_path: Path, feed_entries: list[dict]
    ) -> tuple[dict[str, dict], list[str]]:
        """Call batch-feed-detail CLI once for all feeds in a single browser session."""
        feeds_json = json.dumps(feed_entries, ensure_ascii=False)
        cmd = [
            sys.executable,
            str(cli_path),
            "batch-feed-detail",
            "--feeds-json", feeds_json,
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._skills_dir / "scripts"),
            )
            timeout = 30 + 10 * len(feed_entries)
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)

            if proc.returncode != 0:
                logger.warning("batch-feed-detail exited with code %d", proc.returncode)
                return {}, [e.get("feedId", "") for e in feed_entries]

            batch = json.loads(stdout.decode("utf-8"))
            return batch.get("results", {}), batch.get("errors", [])
        except Exception as e:
            logger.warning("batch-feed-detail failed: %s", e)
            return {}, [e_item.get("feedId", "") for e_item in feed_entries]

    async def _extract_content_from_detail(self, detail: dict) -> str:
        """Extract rich text from a feed detail dict (title + desc + OCR + comments)."""
        note = detail.get("note", {})
        title = note.get("title", "")
        desc = note.get("desc", "")

        parts = []
        if title:
            parts.append(title)
        if desc:
            parts.append(desc)

        image_list = note.get("imageList", [])
        image_urls = [
            img.get("urlDefault", "") or img.get("url", "")
            for img in image_list
            if img.get("urlDefault") or img.get("url")
        ]
        if image_urls:
            ocr_text = await self._ocr_images(image_urls)
            if ocr_text:
                parts.append(f"[图片内容]\n{ocr_text}")

        comments = detail.get("comments", [])
        if isinstance(comments, dict):
            comments = comments.get("list", [])
        for c in comments[:5]:
            c_text = c.get("content", "")
            if c_text and len(c_text) > 10:
                parts.append(f"[评论] {c_text}")

        return "\n\n".join(parts)

    async def _ocr_images(self, image_urls: list[str]) -> str:
        """Download images and extract text via RapidOCR (local CPU inference).

        No upper limit on image count — Xiaohongshu allows up to 18 per post.
        Each image is processed independently; single failures are skipped.
        """
        global _ocr_engine
        if _ocr_engine is None:
            try:
                _ocr_engine = _get_ocr_engine()
            except Exception as e:
                logger.warning("RapidOCR init failed: %s", e)
                return ""

        loop = asyncio.get_event_loop()
        all_texts: list[str] = []

        for i, url in enumerate(image_urls):
            if not url:
                continue
            try:
                img_data = await loop.run_in_executor(
                    None, self._download_image, url
                )
                if not img_data:
                    continue

                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
                    tmp.write(img_data)
                    tmp.flush()
                    result = await loop.run_in_executor(
                        None, _ocr_engine, tmp.name
                    )

                if result and result.txts:
                    page_text = "\n".join(result.txts)
                    all_texts.append(page_text)
                    logger.debug("OCR image %d/%d: %d lines", i + 1, len(image_urls), len(result.txts))
            except Exception as e:
                logger.debug("OCR failed for image %d: %s", i + 1, e)

        if all_texts:
            logger.info("OCR extracted text from %d/%d images", len(all_texts), len(image_urls))
        return "\n\n".join(all_texts)

    @staticmethod
    def _download_image(url: str) -> bytes | None:
        """Download a single image with timeout."""
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://www.xiaohongshu.com/",
            })
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.read()
        except Exception:
            return None

    @staticmethod
    def _extract_likes(feed: dict) -> int:
        interact = feed.get("interactInfo", {})
        raw = interact.get("likedCount", "0")
        if isinstance(raw, int):
            return raw
        text = str(raw).strip()
        if not text:
            return 0
        try:
            if "万" in text:
                return int(float(text.replace("万", "")) * 10000)
            return int(text)
        except ValueError:
            return 0

    async def _fallback_search(
        self, query: str, max_results: int
    ) -> list[SearchResult]:
        """Plan B: Generic web search fallback."""
        import urllib.parse
        import urllib.request

        search_query = urllib.parse.quote(f"小红书 {query} 面试")
        url = f"https://www.bing.com/search?q={search_query}&count={max_results}"

        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"
        })

        results = []
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
                snippets = re.findall(r'<p>(.*?)</p>', html)
                for i, snippet in enumerate(snippets[:max_results]):
                    clean = re.sub(r'<[^>]+>', '', snippet).strip()
                    if clean and len(clean) > 20:
                        results.append(SearchResult(
                            title=f"搜索结果 {i + 1}",
                            content=clean,
                            url=f"bing_search_{i}",
                            likes=0,
                        ))
        except Exception as e:
            logger.error("Fallback search failed: %s", e)

        logger.info("Fallback search: %d results for '%s'", len(results), query)
        return results

    def get_tool_schema(self) -> dict:
        """Return MCP-style tool descriptor."""
        return {
            "name": "search_xiaohongshu",
            "description": "Search Xiaohongshu (小红书) for AI/LLM interview questions and answers",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        }
