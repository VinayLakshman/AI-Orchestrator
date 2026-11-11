from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class KnowledgeRetrieveRequest(BaseModel):
    question: str
    top_k: int = 6
    candidate_limit: int = 12
    neighbor_window: int = 1
    repository: str | None = None
    branch: str | None = None


class KnowledgeHit(BaseModel):
    repository: str
    branch: str
    commit: str
    path: str
    language: str
    chunk_index: int
    chunk_count: int
    score: float | None = None
    content: str
    start_line: int | None = None
    end_line: int | None = None
    url: str | None = None


class KnowledgeRetrieveResponse(BaseModel):
    question: str
    intent: str
    embedding_time: float
    search_time: float
    rerank_time: float
    expansion_time: float
    total_time: float
    context: str | None = None
    grounded: bool = False
    confidence: float = 0.0
    retrieval_reason: str = ""
    primary_hits: list[KnowledgeHit] = Field(default_factory=list)
    expanded_hits: list[KnowledgeHit] = Field(default_factory=list)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    repository: str | None = None
    branch: str | None = None
    top_k: int = Field(default=6, ge=1, le=50)


class AskRequest(SearchRequest):
    include_context: bool = False


class Citation(BaseModel):
    repository: str
    branch: str
    commit: str
    path: str
    chunk_index: int
    score: float | None = None
    start_line: int | None = None
    end_line: int | None = None
    url: str | None = None


class AskResponse(BaseModel):
    answer: str
    grounded: bool
    confidence: float
    citations: list[Citation] = Field(default_factory=list)
    context: str | None = None
    model: str | None = None
    retrieval: dict[str, Any] = Field(default_factory=dict)

