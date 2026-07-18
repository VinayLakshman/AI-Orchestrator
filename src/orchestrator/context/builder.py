from __future__ import annotations

import json
import re
from typing import Any

from ..common.enums import ChatRole
from ..logging import get_logger
from ..models.chat import ChatMessage
from ..models.evidence import (
    EvidenceLedger,
)
from ..models.state import OrchestratorState, RequestState

logger = get_logger(__name__)

CONVERSATION_HISTORY_TOKEN_BUDGET = 2400


def estimate_text_tokens(text: str | None) -> int:
    value = (text or "").strip()
    if not value:
        return 0
    return max(1, len(re.findall(r"\S+", value)))


def _normalize_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


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


def _compact_lines(text: str | None, *, max_items: int, max_chars: int) -> list[str]:
    lines = [
        _truncate(line.strip(" -•\t"), max_chars)
        for line in str(text or "").splitlines()
        if line.strip()
    ]
    deduped, _ = _dedupe_text_items(lines)
    return deduped[:max_items]


def _message_role(message: dict[str, Any] | ChatMessage) -> str:
    if isinstance(message, ChatMessage):
        return str(message.role.value)
    return str(message.get("role") or "")


def _message_content_text(message: dict[str, Any] | ChatMessage) -> str:
    content: Any
    if isinstance(message, ChatMessage):
        content = message.content
    else:
        content = message.get("content")

    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                for key in ("text", "content", "url"):
                    value = part.get(key)
                    if isinstance(value, str) and value.strip():
                        parts.append(value.strip())
                        break
        return "\n".join(parts).strip()
    if content is None:
        return ""
    return str(content).strip()


def _message_token_count(message: dict[str, Any] | ChatMessage) -> int:
    return estimate_text_tokens(_message_content_text(message))


def _message_sequence(messages: list[dict[str, Any] | ChatMessage]) -> list[str]:
    return [_message_role(message) for message in messages]


def _build_conversation_history(
    messages: list[dict[str, Any] | ChatMessage] | None,
) -> tuple[list[ChatMessage], str, dict[str, Any]]:
    raw_messages = list(messages or [])
    if not raw_messages:
        raise ValueError("Conversation is empty")

    role_sequence = _message_sequence(raw_messages)

    latest_user_index: int | None = None
    for index in range(len(raw_messages) - 1, -1, -1):
        message = raw_messages[index]
        role = message.role if isinstance(message, ChatMessage) else message.get("role")
        if role == ChatRole.USER.value or role == ChatRole.USER:
            latest_user_index = index
            break

    if latest_user_index is None:
        raise ValueError("Conversation contains no user message")

    latest_user_text = _message_content_text(raw_messages[latest_user_index])

    history_messages = list(raw_messages[:latest_user_index])

    history_token_count = sum(_message_token_count(message) for message in history_messages)

    truncated = False
    while history_messages and history_token_count > CONVERSATION_HISTORY_TOKEN_BUDGET:
        history_messages.pop(0)
        history_token_count = sum(_message_token_count(message) for message in history_messages)
        truncated = True

    history_chat_messages = [
        ChatMessage.model_validate(message) for message in history_messages
    ]

    conversation_info = {
        "total_message_count": len(raw_messages),
        "role_sequence": role_sequence,
        "latest_user_index": latest_user_index,
        "latest_user_message_length": len(latest_user_text),
        "history_token_count": history_token_count,
        "truncation_occurred": truncated,
        "latest_user_survived_truncation": True,
        "history_message_count": len(history_chat_messages),
    }

    return history_chat_messages, latest_user_text, conversation_info


def _build_evidence_sources(evidence: EvidenceLedger) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []

    if evidence.repository:
        sources.append(
            {
                "type": "knowledge",
                "repository": evidence.repository.repository,
                "branch": evidence.repository.branch,
                "commit": evidence.repository.commit,
                "question": evidence.repository.question,
                "retrieval_reason": evidence.repository.retrieval_reason,
                "confidence": evidence.repository.confidence,
                "summary": evidence.repository.context,
                "hit_count": evidence.repository.hit_count,
                "evidence": [
                    _evidence_to_text(hit)
                    for hit in (evidence.repository.primary_hits or [])
                    if _evidence_to_text(hit).strip()
                ][:8],
                "extended_evidence": [
                    _evidence_to_text(hit)
                    for hit in (evidence.repository.expanded_hits or [])
                    if _evidence_to_text(hit).strip()
                ][:4],
                "metadata": evidence.repository.metadata,
            }
        )

    if evidence.web:
        sources.append(
            {
                "type": "web",
                "query": evidence.web.query,
                "confidence": evidence.web.confidence,
                "summary": evidence.web.summary,
                "evidence": [
                    _evidence_to_text(item)
                    for item in (evidence.web.snippets or [])
                    if _evidence_to_text(item).strip()
                ][:8],
                "urls": evidence.web.urls,
                "results": [
                    item if isinstance(item, dict) else _evidence_to_text(item)
                    for item in (evidence.web.results or [])
                ],
                "metadata": evidence.web.metadata,
            }
        )

    if evidence.vision:
        sources.append(
            {
                "type": "vision",
                "task": evidence.vision.task,
                "confidence": evidence.vision.confidence,
                "summary": evidence.vision.summary,
                "observations": evidence.vision.observations,
                "evidence": [
                    evidence.vision.extracted_text,
                    *_compact_lines(
                        evidence.vision.extracted_text,
                        max_items=5,
                        max_chars=180,
                    ),
                ],
                "detected_objects": evidence.vision.detected_objects,
                "metadata": evidence.vision.metadata,
            }
        )

    if evidence.code:
        sources.append(
            {
                "type": "code",
                "language": evidence.code.language,
                "task": evidence.code.task,
                "confidence": evidence.code.confidence,
                "summary": evidence.code.summary,
                "evidence": [
                    evidence.code.generated_code,
                    evidence.code.explanation,
                    *evidence.code.tests,
                    *evidence.code.warnings,
                ],
                "files": evidence.code.files,
                "metadata": evidence.code.metadata,
            }
        )

    if evidence.tools:
        sources.append(
            {
                "type": "tool",
                "evidence": [
                    json.dumps(
                        execution.model_dump(exclude_none=True),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    for execution in evidence.tools.executions
                ],
                "metadata": evidence.tools.metadata,
            }
        )

    if evidence.reasoning:
        sources.append(
            {
                "type": "reasoning",
                "summary": evidence.reasoning.summary,
                "conclusions": evidence.reasoning.conclusions,
                "assumptions": evidence.reasoning.assumptions,
                "evidence": [
                    evidence.reasoning.summary,
                    *evidence.reasoning.conclusions,
                    *evidence.reasoning.assumptions,
                ],
                "metadata": evidence.reasoning.metadata,
            }
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
        "sources": _build_evidence_sources(evidence),
    }


def build_controller_messages(
    *,
    system_prompt: str,
    messages: list[dict[str, Any] | ChatMessage] | None = None,
    request_context: str = "",
    structured_context: str = "",
    additional_context: str = "",
) -> list[ChatMessage]:
    history_messages, latest_user_message, conversation_info = _build_conversation_history(messages)
    outgoing: list[ChatMessage] = [ChatMessage(role=ChatRole.SYSTEM, content=system_prompt)]

    if request_context:
        outgoing.append(
            ChatMessage(
                role=ChatRole.SYSTEM,
                metadata={"source": "normalized_request"},
                content=request_context,
            )
        )

    if structured_context:
        outgoing.append(
            ChatMessage(
                role=ChatRole.SYSTEM,
                metadata={"source": "structured_context"},
                content=structured_context,
            )
        )

    if additional_context:
        outgoing.append(
            ChatMessage(
                role=ChatRole.SYSTEM,
                metadata={"source": "additional_context"},
                content=additional_context,
            )
        )

    outgoing.extend(history_messages)
    outgoing.append(ChatMessage(role=ChatRole.USER, content=latest_user_message))

    logger.debug(
        "conversation_assembly %s",
        json.dumps(conversation_info, sort_keys=True, default=str),
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
