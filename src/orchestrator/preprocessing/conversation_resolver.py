from __future__ import annotations

import json
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from pydantic import BaseModel, Field

from ..clients.ollama import OllamaClient
from ..common.enums import ChatRole
from ..logging import get_logger
from ..models.chat import ChatMessage
from ..models.manager import ModelManager
from ..models.state import RequestState
from ..settings import Settings
from .context_builder import build_resolver_context

logger = get_logger(__name__)

RESOLVER_CONFIDENCE_THRESHOLD = 0.85

RESOLVER_SYSTEM_PROMPT = """You are responsible for resolving conversational references.

Given recent conversation history and the latest user message, rewrite the latest user message into a fully self-contained request.

Rules:
- Preserve user intent.
- Resolve references such as it, that, those, this, again, same, previous, earlier, second, first, above.
- Resolve references to previous assistant responses.
- Resolve references to previous user questions.
- Do not answer the question.
- Do not classify intent.
- Do not invoke tools.
- Do not perform searches.
- Only rewrite the latest request.
- If the request is already self-contained, return it unchanged.
- Return structured JSON only.
"""


class ConversationResolution(BaseModel):
    resolved_query: str = ""
    is_followup: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


def _extract_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()

    start = text.find("{")
    end = text.rfind("}")
    candidate = text[start : end + 1] if start != -1 and end != -1 and end > start else text
    try:
        return json.loads(candidate)
    except Exception:
        return {}


def _normalize_resolution(parsed: dict[str, Any], *, original_query: str) -> ConversationResolution:
    resolved_query = str(
        parsed.get("resolved_query")
        or parsed.get("resolved")
        or parsed.get("query")
        or ""
    ).strip()
    if not resolved_query:
        resolved_query = original_query.strip()

    try:
        confidence = float(parsed.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0

    return ConversationResolution(
        resolved_query=resolved_query,
        is_followup=bool(parsed.get("is_followup")),
        confidence=confidence,
    )


def _build_resolver_messages(
    *,
    context_history: list[ChatMessage],
    latest_user_message: str,
) -> list[ChatMessage]:
    messages: list[ChatMessage] = [ChatMessage(role=ChatRole.SYSTEM, content=RESOLVER_SYSTEM_PROMPT)]
    messages.extend(context_history)
    if latest_user_message:
        messages.append(ChatMessage(role=ChatRole.USER, content=latest_user_message))
    return messages


def _rewrite_latest_user_message(messages: list[ChatMessage], resolved_query: str) -> list[ChatMessage]:
    updated = list(messages)
    for index in range(len(updated) - 1, -1, -1):
        if updated[index].role == ChatRole.USER:
            updated[index] = updated[index].model_copy(update={"content": resolved_query})
            break
    return updated


@dataclass(slots=True)
class ResolverResult:
    request: RequestState
    resolution: ConversationResolution
    resolution_time_ms: float
    applied: bool


async def resolve_conversation_context(
    request: RequestState,
    *,
    settings: Settings,
    model_manager: ModelManager,
    ollama_client: OllamaClient,
) -> ResolverResult:
    original_query = str(request.original_query or request.user_message or "").strip()
    resolved_query = original_query
    resolution = ConversationResolution(resolved_query=original_query, confidence=0.0, is_followup=False)

    context = build_resolver_context(request.messages)
    started = perf_counter()

    if not original_query or not context.has_history:
        elapsed_ms = (perf_counter() - started) * 1000.0
        updated = request.model_copy(
            update={
                "original_query": original_query,
                "resolved_query": original_query,
                "user_message": original_query,
                "messages": _rewrite_latest_user_message(list(request.messages), original_query),
                "is_followup": False,
                "followup_confidence": 0.0,
                "metadata": {
                    **request.metadata,
                    "conversation_resolution": {
                        "original_query": original_query,
                        "resolved_query": original_query,
                        "model_resolved_query": original_query,
                        "is_followup": False,
                        "model_is_followup": False,
                        "confidence": 0.0,
                        "applied": False,
                        "resolution_time_ms": elapsed_ms,
                    },
                },
            }
        )
        logger.debug(
            "conversation_resolver original_query=%r resolved_query=%r is_followup=%s confidence=%.3f resolution_time_ms=%.2f",
            original_query,
            original_query,
            False,
            0.0,
            elapsed_ms,
        )
        return ResolverResult(request=updated, resolution=resolution, resolution_time_ms=elapsed_ms, applied=False)

    messages = _build_resolver_messages(
        context_history=context.history_messages,
        latest_user_message=context.latest_user_message or original_query,
    )

    try:
        response = await ollama_client.chat(
            model=model_manager.controller().name,
            messages=messages,
            temperature=0.0,
            max_tokens=256,
            stream=False,
            response_format="json",
            keep_alive=settings.controller_keep_alive,
        )
        raw_content = str(response.content or response.raw or "").strip()
        parsed = _extract_json_object(raw_content)
        if not isinstance(parsed, dict):
            parsed = {}
        resolution = _normalize_resolution(parsed, original_query=original_query)
    except Exception:
        logger.exception("conversation_resolver_failed")
        resolution = ConversationResolution(resolved_query=original_query, is_followup=False, confidence=0.0)

    applied = bool(
        resolution.confidence >= RESOLVER_CONFIDENCE_THRESHOLD
        and resolution.resolved_query.strip()
    )
    resolved_query = resolution.resolved_query.strip() if applied else original_query
    is_followup = applied and resolved_query != original_query
    elapsed_ms = (perf_counter() - started) * 1000.0

    updated = request.model_copy(
        update={
            "original_query": original_query,
            "resolved_query": resolved_query,
            "user_message": resolved_query,
            "messages": _rewrite_latest_user_message(list(request.messages), resolved_query),
            "is_followup": is_followup,
            "followup_confidence": float(resolution.confidence or 0.0),
            "metadata": {
                **request.metadata,
                "conversation_resolution": {
                    "original_query": original_query,
                    "resolved_query": resolved_query,
                    "model_resolved_query": resolution.resolved_query.strip(),
                    "is_followup": is_followup,
                    "model_is_followup": bool(resolution.is_followup),
                    "confidence": float(resolution.confidence or 0.0),
                    "applied": applied,
                    "resolution_time_ms": elapsed_ms,
                },
            },
        }
    )

    logger.debug(
        "conversation_resolver original_query=%r resolved_query=%r is_followup=%s confidence=%.3f resolution_time_ms=%.2f",
        original_query,
        resolved_query,
        is_followup,
        float(resolution.confidence or 0.0),
        elapsed_ms,
    )

    return ResolverResult(
        request=updated,
        resolution=resolution,
        resolution_time_ms=elapsed_ms,
        applied=applied,
    )
