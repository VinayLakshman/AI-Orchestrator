from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..models.knowledge import KnowledgeHit
from ..settings import Settings


@dataclass(slots=True)
class RetrievalValidationResult:
    grounded: bool
    context: str
    score: float
    hit_count: int
    reason: str
    metadata: dict[str, Any]


def render_knowledge_context(chunks: list[KnowledgeHit]) -> str:
    if not chunks:
        return ""

    lines: list[str] = ["Retrieved knowledge context:"]
    for idx, chunk in enumerate(chunks, start=1):
        score = f" (score={chunk.score:.3f})" if isinstance(chunk.score, float) else ""
        source = f" source={chunk.repository}" if chunk.repository else ""
        lines.append(f"[{idx}]{score}{source}")
        lines.append(chunk.content.strip())
        lines.append("")
    return "\n".join(lines).strip()


def validate_retrieval(
    *,
    knowledge_context: str,
    knowledge_hits: list[KnowledgeHit],
    retrieval_stats: dict[str, Any],
    settings: Settings,
    render_context: Any,
) -> RetrievalValidationResult:
    context = (knowledge_context or "").strip()

    if not context and knowledge_hits:
        context = render_context(knowledge_hits)

    hit_count = len(knowledge_hits)
    score = max((hit.score for hit in knowledge_hits), default=0.0)

    min_score = getattr(settings, "knowledge_min_score", 0.55)
    min_hits = getattr(settings, "knowledge_min_hits", 1)

    grounded = bool(context) and hit_count >= min_hits and score >= min_score

    reasons: list[str] = []
    if not context:
        reasons.append("no usable retrieval context")
    if hit_count < min_hits:
        reasons.append(f"only {hit_count} retrieved chunks")
    if hit_count and score < min_score:
        reasons.append(f"best score {score:.3f} below threshold {min_score:.2f}")

    return RetrievalValidationResult(
        grounded=grounded,
        context=context,
        score=score,
        hit_count=hit_count,
        reason=", ".join(reasons),
        metadata=retrieval_stats,
    )
