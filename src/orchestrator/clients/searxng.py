from __future__ import annotations

import time
from urllib.parse import urlsplit, urlunsplit

import httpx

from ..logging import get_logger
from ..models.web import SearchResult, WebSearchResult
from ..settings import Settings

logger = get_logger(__name__)


def normalize_query(query: str) -> str:
    return " ".join((query or "").split()).strip()


class SearXNGClient:
    """Small transport-only client for SearXNG's JSON search endpoint."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self.client = client

    async def search(self, query: str, *, max_results: int | None = None) -> WebSearchResult:
        normalized = normalize_query(query)
        logger.debug("web_search_normalized_query query=%r", normalized)
        if not normalized:
            return WebSearchResult(query="")

        limit = max(1, max_results or self.settings.web_search_max_results)
        params = {
            "q": normalized,
            "format": "json",
            "language": self.settings.web_search_language,
            "safesearch": self.settings.web_search_safesearch,
            "categories": self.settings.web_search_categories,
        }
        logger.debug("web_search_request_parameters %s", {**params, "q": "<redacted-query>"})
        started = time.perf_counter()
        close_client = False
        client = self.client
        if client is None:
            client = httpx.AsyncClient(
                base_url=self.settings.web_search_url,
                timeout=httpx.Timeout(self.settings.web_search_timeout_s),
            )
            close_client = True
        try:
            response = await client.get("/search", params=params)
            response.raise_for_status()
            payload = response.json()
            raw_results = payload.get("results", []) if isinstance(payload, dict) else []
            results: list[SearchResult] = []
            seen: set[str] = set()
            for raw in raw_results if isinstance(raw_results, list) else []:
                if not isinstance(raw, dict):
                    continue
                title = str(raw.get("title") or "").strip()
                url = str(raw.get("url") or raw.get("link") or "").strip()
                snippet = str(raw.get("content") or raw.get("snippet") or "").strip()
                if not title or not url:
                    continue
                parts = urlsplit(url)
                key = urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), parts.query, ""))
                if not key or key in seen:
                    continue
                seen.add(key)
                score = raw.get("score")
                results.append(SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    engine=str(raw.get("engine") or "").strip(),
                    score=float(score) if isinstance(score, (int, float)) else None,
                ))
                if len(results) >= limit:
                    break
            duration = int((time.perf_counter() - started) * 1000)
            logger.info("web_search_completed search_duration_ms=%d results_returned=%d", duration, len(results))
            logger.debug("web_search_deduplication returned_titles=%s returned_urls=%s", [r.title for r in results], [r.url for r in results])
            return WebSearchResult(query=normalized, search_time_ms=duration, results=results)
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            duration = int((time.perf_counter() - started) * 1000)
            logger.warning("web_search_failed search_duration_ms=%d error=%s", duration, str(exc)[:160])
            return WebSearchResult(query=normalized, search_time_ms=duration, error=str(exc)[:160])
        finally:
            if close_client:
                await client.aclose()


__all__ = ["SearXNGClient", "normalize_query"]
