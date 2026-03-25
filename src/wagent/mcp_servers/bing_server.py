"""MCP Server for Bing Web Search.

Uses requests + HTML parsing (no API key required).
Exposes the same MCP tool interface as XiaohongshuMCPServer for consistency.
"""

from __future__ import annotations

import asyncio
import logging
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


@dataclass
class BingResult:
    title: str
    snippet: str
    url: str
    is_pdf: bool = False


class BingMCPServer:
    """MCP-style server wrapping Bing Web Search via HTML scraping."""

    async def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        """MCP tool: search_bing.

        Returns list of dicts with keys: title, snippet, url, is_pdf.
        """
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, self._scrape_bing, query, max_results
        )
        return [
            {
                "title": r.title,
                "snippet": r.snippet,
                "url": r.url,
                "is_pdf": r.is_pdf,
            }
            for r in results
        ]

    def _scrape_bing(self, query: str, max_results: int) -> list[BingResult]:
        """Synchronous Bing scraping."""
        encoded = urllib.parse.quote_plus(query)
        url = f"https://www.bing.com/search?q={encoded}&count={min(max_results * 2, 30)}"

        req = urllib.request.Request(url, headers={
            "User-Agent": USER_AGENT,
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        })

        results: list[BingResult] = []
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
                results = self._parse_bing_html(html, max_results)
        except Exception as e:
            logger.error("Bing search failed: %s", e)

        logger.info("Bing search: %d results for '%s'", len(results), query)
        return results

    @staticmethod
    def _parse_bing_html(html: str, max_results: int) -> list[BingResult]:
        """Extract search results from Bing HTML response."""
        results: list[BingResult] = []

        li_pattern = re.compile(
            r'<li[^>]*class="b_algo"[^>]*>(.*?)</li>',
            re.DOTALL,
        )
        blocks = li_pattern.findall(html)

        for block in blocks:
            if len(results) >= max_results:
                break

            href_match = re.search(r'<a[^>]+href="(https?://[^"]+)"', block)
            title_match = re.search(r'<a[^>]+>(.*?)</a>', block, re.DOTALL)
            snippet_match = re.search(
                r'<p[^>]*>(.*?)</p>|<div[^>]*class="b_caption"[^>]*>.*?<p>(.*?)</p>',
                block,
                re.DOTALL,
            )

            if not href_match:
                continue

            url = href_match.group(1)
            title = re.sub(r'<[^>]+>', '', title_match.group(1)).strip() if title_match else ""
            snippet = ""
            if snippet_match:
                raw = snippet_match.group(1) or snippet_match.group(2) or ""
                snippet = re.sub(r'<[^>]+>', '', raw).strip()

            if not title and not snippet:
                continue

            is_pdf = url.lower().endswith(".pdf") or "application/pdf" in block.lower()

            results.append(BingResult(
                title=title,
                snippet=snippet,
                url=url,
                is_pdf=is_pdf,
            ))

        return results

    def get_tool_schema(self) -> dict:
        """Return MCP-style tool descriptor."""
        return {
            "name": "search_bing",
            "description": "Search the web via Bing for technical content, documentation, and resources",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        }
