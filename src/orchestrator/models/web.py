from __future__ import annotations

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    title: str = ""
    url: str = ""
    snippet: str = ""
    engine: str = ""
    score: float | None = None


class WebSearchResult(BaseModel):
    query: str = ""
    search_time_ms: int = 0
    results: list[SearchResult] = Field(default_factory=list)
    error: str = ""


__all__ = ["SearchResult", "WebSearchResult"]
