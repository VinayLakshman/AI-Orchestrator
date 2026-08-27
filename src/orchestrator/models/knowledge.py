from __future__ import annotations

from pydantic import BaseModel, Field


class KnowledgeRetrieveRequest(BaseModel):
    question: str
    top_k: int = 6
    candidate_limit: int = 12
    neighbor_window: int = 1


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

