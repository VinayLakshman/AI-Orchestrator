from __future__ import annotations

from ..clients.searxng import SearXNGClient, normalize_query
from ..models.web import WebSearchResult


class WebSpecialist:
    """Coordinates web evidence retrieval without generating or summarizing prose."""

    def __init__(self, client: SearXNGClient) -> None:
        self.client = client

    async def retrieve(
        self,
        query: str,
        *,
        cached: WebSearchResult | None = None,
        max_results: int,
    ) -> WebSearchResult:
        normalized = normalize_query(query)
        if cached is not None and cached.query == normalized:
            return cached
        return await self.client.search(normalized, max_results=max_results)


__all__ = ["WebSpecialist"]
