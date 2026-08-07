"""Context rendering and compatibility wrappers.

This module retains the context *rendering* helpers (structured context,
request context, finalize context) and exposes thin compatibility wrappers
for the historical message-builder names. All conversation construction now
delegates to ``orchestrator.context.assembler`` and all conversation parsing
delegates to ``orchestrator.context.parser``.
"""

from __future__ import annotations

import json
from typing import Any

from ..logging import get_logger
from ..models.chat import ChatMessage
from ..models.evidence import (
    EvidenceLedger,
)
from ..models.state import OrchestratorState, RequestState
from .assembler import build_conversation
from .parser import estimate_text_tokens, split_conversation

logger = get_logger(__name__)


def _evidence_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("content", "snippet", "text", "summary", "title", "url", "evidence"):
            raw = value.get(key)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, list):
        parts = [_evidence_to_text(item) for item in value]
        return "\n".join(part for part in parts if part.strip())
    return str(value).strip()


def _dump_if_possible(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if isinstance(value, dict):
        return {key: item for key, item in value.items() if item is not None}
    return value


def _first_text(values: list[Any], *, limit: int = 220) -> str:
    for value in values:
        text = _truncate(_evidence_to_text(value), limit)
        if text:
            return text
    return ""


def _normalize_text(text: str | None) -> str:
    return " ".join(str(text or "").split()).strip()


def _truncate(text: str, limit: int = 220) -> str:
    cleaned = _normalize_text(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 1)].rstrip() + "…"


def _dedupe_text_items(values: list[str]) -> tuple[list[str], int]:
    seen: set[str] = set()
    deduped: list[str] = []
    removed = 0
    for value in values:
        item = _normalize_text(value)
        if not item:
            continue
        key = item.lower()
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        deduped.append(item)
    return deduped, removed


def _compact_lines(text: str | None, *, max_items: int, max_chars: int) -> list[str]:
    lines = [
        _truncate(line.strip(" -•\t"), max_chars)
        for line in str(text or "").splitlines()
        if line.strip()
    ]
    deduped, _ = _dedupe_text_items(lines)
    return deduped[:max_items]


def _structured_source_entry(
    *,
    source_type: str,
    confidence: float,
    summary: str,
    relevant_excerpts: list[str],
    supporting_facts: list[Any],
    provenance: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "type": source_type,
        "source": provenance.get("source", source_type),
        "status": "validated",
        "confidence": confidence,
        "summary": summary,
        "relevant_excerpts": [item for item in relevant_excerpts if item.strip()][:8],
        "supporting_facts": [
            _dump_if_possible(item)
            for item in supporting_facts
            if item is not None
        ][:8],
        "provenance": provenance,
        "metadata": metadata,
    }


def _build_validated_evidence_sources(evidence: EvidenceLedger) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []

    if evidence.repository:
        primary_hits = [
            _dump_if_possible(hit)
            for hit in (evidence.repository.primary_hits or [])
            if hit is not None
        ]
        expanded_hits = [
            _dump_if_possible(hit)
            for hit in (evidence.repository.expanded_hits or [])
            if hit is not None
        ]
        relevant_excerpts = [
            _first_text([hit], limit=260)
            for hit in primary_hits[:8]
            if _first_text([hit], limit=260).strip()
        ]
        if evidence.repository.context:
            relevant_excerpts.insert(0, _truncate(evidence.repository.context, 400))

        sources.append(
            _structured_source_entry(
                source_type="knowledge",
                confidence=float(evidence.repository.confidence or 0.0),
                summary=str(evidence.repository.context or "").strip(),
                relevant_excerpts=relevant_excerpts,
                supporting_facts=primary_hits + expanded_hits,
                provenance={
                    "source": "Knowledge Base",
                    "repository": evidence.repository.repository,
                    "branch": evidence.repository.branch,
                    "commit": evidence.repository.commit,
                    "question": evidence.repository.question,
                    "retrieval_reason": evidence.repository.retrieval_reason,
                },
                metadata=evidence.repository.metadata,
            )
        )

    if evidence.web:
        results = [
            _dump_if_possible(item)
            for item in (evidence.web.results or [])
            if item is not None
        ]
        relevant_excerpts = [
            _truncate(item, 260)
            for item in (evidence.web.snippets or [])
            if _truncate(item, 260).strip()
        ]
        if evidence.web.summary:
            relevant_excerpts.insert(0, _truncate(evidence.web.summary, 400))

        sources.append(
            _structured_source_entry(
                source_type="web",
                confidence=float(evidence.web.confidence or 0.0),
                summary=str(evidence.web.summary or "").strip(),
                relevant_excerpts=relevant_excerpts,
                supporting_facts=results,
                provenance={
                    "source": "Web Search",
                    "query": evidence.web.query,
                },
                metadata=evidence.web.metadata,
            )
        )

    if evidence.vision:
        observations = [str(item).strip() for item in (evidence.vision.observations or []) if str(item).strip()]
        relevant_excerpts = [
            _truncate(item, 260)
            for item in [
                evidence.vision.summary,
                evidence.vision.context,
                evidence.vision.extracted_text,
                *observations,
            ]
            if _truncate(item, 260).strip()
        ]

        sources.append(
            _structured_source_entry(
                source_type="vision",
                confidence=float(evidence.vision.confidence or 0.0),
                summary=str(evidence.vision.summary or "").strip(),
                relevant_excerpts=relevant_excerpts,
                supporting_facts=[
                    *observations,
                    *[str(item).strip() for item in (evidence.vision.detected_objects or []) if str(item).strip()],
                ],
                provenance={
                    "source": "Vision",
                    "task": evidence.vision.task,
                },
                metadata=evidence.vision.metadata,
            )
        )

    if evidence.code:
        relevant_excerpts = [
            _truncate(item, 260)
            for item in [
                evidence.code.summary,
                evidence.code.explanation,
                evidence.code.generated_code,
                *evidence.code.tests,
                *evidence.code.warnings,
            ]
            if _truncate(item, 260).strip()
        ]

        sources.append(
            _structured_source_entry(
                source_type="code",
                confidence=float(evidence.code.confidence or 0.0),
                summary=str(evidence.code.summary or "").strip(),
                relevant_excerpts=relevant_excerpts,
                supporting_facts=[
                    *[str(item).strip() for item in (evidence.code.files or []) if str(item).strip()],
                    *[str(item).strip() for item in (evidence.code.tests or []) if str(item).strip()],
                    *[str(item).strip() for item in (evidence.code.warnings or []) if str(item).strip()],
                ],
                provenance={
                    "source": "Code Analysis",
                    "language": evidence.code.language,
                    "task": evidence.code.task,
                },
                metadata=evidence.code.metadata,
            )
        )

    if evidence.tools:
        executions = [
            _dump_if_possible(execution)
            for execution in (evidence.tools.executions or [])
            if execution is not None
        ]
        relevant_excerpts = [
            _truncate(_evidence_to_text(execution), 260)
            for execution in executions
            if _truncate(_evidence_to_text(execution), 260).strip()
        ]

        sources.append(
            _structured_source_entry(
                source_type="tool",
                confidence=1.0,
                summary="Tool executions available.",
                relevant_excerpts=relevant_excerpts,
                supporting_facts=executions,
                provenance={
                    "source": "Future MCP tools",
                },
                metadata=evidence.tools.metadata,
            )
        )

    if evidence.reasoning:
        relevant_excerpts = [
            _truncate(item, 260)
            for item in [
                evidence.reasoning.summary,
                *evidence.reasoning.conclusions,
                *evidence.reasoning.assumptions,
            ]
            if _truncate(item, 260).strip()
        ]

        sources.append(
            _structured_source_entry(
                source_type="reasoning",
                confidence=1.0,
                summary=str(evidence.reasoning.summary or "").strip(),
                relevant_excerpts=relevant_excerpts,
                supporting_facts=[
                    *[str(item).strip() for item in (evidence.reasoning.conclusions or []) if str(item).strip()],
                    *[str(item).strip() for item in (evidence.reasoning.assumptions or []) if str(item).strip()],
                ],
                provenance={
                    "source": "Reasoning",
                },
                metadata=evidence.reasoning.metadata,
            )
        )

    return sources


def render_structured_context(state: OrchestratorState) -> str:
    return json.dumps(
        build_finalize_context(state),
        indent=2,
        ensure_ascii=False,
        default=str,
    )


def render_request_context(request: RequestState) -> str:
    payload = {
        "query": request.user_message,
        "metadata": {
            "message_count": int(request.metadata.get("message_count", 0) or 0),
            "images": int(request.metadata.get("image_count", 0) or 0),
            "files": int(request.metadata.get("file_count", 0) or 0),
            "urls": int(bool(request.metadata.get("contains_urls", False))),
            "code_blocks": int(bool(request.metadata.get("contains_code_blocks", False))),
            "estimated_tokens": int(request.metadata.get("estimated_prompt_tokens", 0) or 0),
        },
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def build_finalize_context(state: OrchestratorState) -> dict[str, Any]:
    request = state.request
    execution = state.execution
    evidence = state.evidence

    question = request.user_message

    execution_summary = {
        "classification": execution.plan.classification,
        "confidence": execution.plan.confidence,
        "requires_repository": execution.plan.requires_repository,
        "requires_web": execution.plan.requires_web,
        "requires_reasoning": execution.plan.requires_reasoning,
        "requires_code": execution.plan.requires_code,
        "requires_tools": execution.plan.requires_tools,
        "requires_vision": execution.plan.requires_vision,
        "execution_queue": [
            step.value if hasattr(step, "value") else str(step)
            for step in execution.plan.execution_queue
        ],
        "current_specialist": (
            execution.runtime.current_specialist.value
            if execution.runtime.current_specialist
            else None
        ),
        "completed": [
            step.value if hasattr(step, "value") else str(step)
            for step in execution.runtime.completed
        ],
        "retry_counts": {
            (
                key.value if hasattr(key, "value") else str(key)
            ): (
                value.model_dump(exclude_none=True)
                if hasattr(value, "model_dump")
                else value
            )
            for key, value in execution.runtime.retries.items()
        },
        "validation": execution.validation.model_dump(exclude_none=True) if execution.validation else None,
    }

    return {
        "question": question,
        "execution": execution_summary,
        "sources": _build_validated_evidence_sources(evidence),
    }


def build_controller_messages(
    *,
    system_prompt: str,
    messages: list[dict[str, Any] | ChatMessage] | None = None,
    request_context: str = "",
    structured_context: str = "",
    additional_context: str = "",
) -> list[ChatMessage]:
    """Build the controller/finalizer conversation.

    Delegates to the assembler. The conversation is parsed to recover history
    and the latest user message, then assembled into a single valid outbound
    conversation.
    """
    history_messages, latest_user_message, conversation_info = split_conversation(
        messages
    )

    outgoing = build_conversation(
        system_prompt=system_prompt,
        request_context=request_context,
        structured_context=structured_context,
        additional_context=additional_context,
        history=history_messages,
        latest_user_message=latest_user_message,
    )

    logger.debug(
        "conversation_assembly %s",
        json.dumps(
            conversation_info,
            sort_keys=True,
            default=str,
        ),
    )

    return outgoing


def build_finalizer_messages(
    *,
    system_prompt: str,
    messages: list[dict[str, Any]] | None = None,
    evidence_context: str = "",
) -> list[ChatMessage]:
    return build_controller_messages(
        system_prompt=system_prompt,
        messages=messages,
        structured_context=evidence_context,
    )
