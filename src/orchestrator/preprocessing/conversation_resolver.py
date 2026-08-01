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
RESOLVER_INTENTS = {
    "NEW_TOPIC",
    "FOLLOW_UP",
    "ELABORATION",
    "MODIFICATION",
    "RETRY",
    "COMPARISON",
    "CORRECTION",
    "CLARIFICATION",
}
RESOLVER_ENTITY_SOURCES = {
    "previous_user_message",
    "previous_assistant_response",
    "recent_conversation",
    "latest_user_message",
    "external_fallback",
}

RESOLVER_SYSTEM_PROMPT = """You are an Intent & Context Resolver.

Your job is to understand the latest request in the context of the recent
conversation, then rewrite the latest user message into a fully self-contained
request.

Rules:
- Conversation is the highest authority.
- Conversation entities always override user metadata, user location, world
  knowledge, and model assumptions.
- Preserve the conversational subject unless the intent is NEW_TOPIC.
- Use recent conversation entities first, then previous assistant responses,
  then previous user requests.
- Resolve references such as it, that, those, this, there, here, them, he,
  she, they, former, latter, first, second, previous, above, again, same.
- Do not answer the question.
- Do not use tools.
- Do not perform searches.
- Do not explain your reasoning.
- Only rewrite the latest request.
- If the request is already self-contained, return it unchanged.
- Intent directly drives the rewrite:
  - NEW_TOPIC: treat the request independently and ignore prior subjects.
  - FOLLOW_UP: preserve the subject and resolve references aggressively.
  - ELABORATION: expand the previous answer while preserving the subject.
  - MODIFICATION: modify the previously generated artifact, not a new topic.
  - RETRY: preserve the original request and adjust execution intent only.
  - COMPARISON: preserve the subject and introduce the comparison target.
  - CORRECTION: correct the previous assistant response.
  - CLARIFICATION: clarify the immediately preceding explanation.
- Return JSON only.

Return these fields:
- intent
- conversation_subject
- resolved_query
- resolved_entities
- entity_confidence
- rewrite_confidence
- confidence
- is_followup

resolved_entities must be an object whose values include resolved_to, source,
and confidence.
"""


class ResolvedEntity(BaseModel):
    resolved_to: str = ""
    source: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ConversationResolution(BaseModel):
    resolved_query: str = ""
    intent: str = "NEW_TOPIC"
    conversation_subject: str = ""
    resolved_entities: dict[str, ResolvedEntity] = Field(default_factory=dict)
    entity_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rewrite_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    is_followup: bool = False


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


def _parse_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed < 0.0:
        return default
    if parsed > 1.0:
        return 1.0
    return parsed


def _normalize_entity(
    value: Any,
    *,
    mention: str,
) -> ResolvedEntity:
    if isinstance(value, dict):
        resolved_to = str(
            value.get("resolved_to")
            or value.get("resolved")
            or value.get("target")
            or value.get("entity")
            or value.get("value")
            or ""
        ).strip()
        source = str(value.get("source") or "").strip()
        confidence = _parse_float(value.get("confidence"), default=0.0)
        if source and source not in RESOLVER_ENTITY_SOURCES:
            source = ""
        return ResolvedEntity(
            resolved_to=resolved_to,
            source=source,
            confidence=confidence,
        )

    resolved_to = str(value or "").strip()
    return ResolvedEntity(resolved_to=resolved_to, source="external_fallback", confidence=0.0)


def _normalize_resolved_entities(raw_entities: Any) -> dict[str, ResolvedEntity]:
    resolved: dict[str, ResolvedEntity] = {}

    if isinstance(raw_entities, dict):
        for key, value in raw_entities.items():
            mention = str(key or "").strip()
            if not mention:
                continue
            resolved[mention] = _normalize_entity(value, mention=mention)
        return resolved

    if isinstance(raw_entities, list):
        for item in raw_entities:
            if not isinstance(item, dict):
                continue
            mention = str(
                item.get("reference")
                or item.get("mention")
                or item.get("source_text")
                or item.get("text")
                or ""
            ).strip()
            if not mention:
                continue
            entity_payload = {
                "resolved_to": item.get("resolved_to")
                or item.get("resolved")
                or item.get("target")
                or item.get("entity")
                or item.get("value")
                or "",
                "source": item.get("source") or "",
                "confidence": item.get("confidence") or 0.0,
            }
            resolved[mention] = _normalize_entity(entity_payload, mention=mention)

    return resolved


def _normalize_intent(parsed: dict[str, Any], *, original_query: str) -> str:
    raw_intent = (
        str(parsed.get("intent") or parsed.get("conversation_intent") or "")
        .strip()
        .upper()
    )
    if raw_intent in RESOLVER_INTENTS:
        return raw_intent
    if str(parsed.get("resolved_query") or "").strip() != original_query:
        return "FOLLOW_UP"
    return "NEW_TOPIC"


def _normalize_resolution(parsed: dict[str, Any], *, original_query: str) -> ConversationResolution:
    resolved_query = str(
        parsed.get("resolved_query")
        or parsed.get("resolved")
        or parsed.get("query")
        or ""
    ).strip()
    if not resolved_query:
        resolved_query = original_query.strip()

    intent = _normalize_intent(parsed, original_query=original_query)
    conversation_subject = str(
        parsed.get("conversation_subject")
        or parsed.get("subject")
        or ""
    ).strip()
    resolved_entities = _normalize_resolved_entities(parsed.get("resolved_entities"))
    entity_confidence = _parse_float(parsed.get("entity_confidence"), default=0.0)
    rewrite_confidence = _parse_float(
        parsed.get("rewrite_confidence")
        if parsed.get("rewrite_confidence") is not None
        else parsed.get("confidence"),
        default=0.0,
    )

    if not rewrite_confidence and parsed.get("confidence") is not None:
        rewrite_confidence = _parse_float(parsed.get("confidence"), default=0.0)

    confidence = rewrite_confidence

    return ConversationResolution(
        resolved_query=resolved_query,
        intent=intent,
        conversation_subject=conversation_subject,
        resolved_entities=resolved_entities,
        entity_confidence=entity_confidence,
        rewrite_confidence=rewrite_confidence,
        confidence=confidence,
        is_followup=bool(parsed.get("is_followup")),
    )


def _structured_context_payload(context: Any, *, original_query: str) -> dict[str, Any]:
    return {
        "original_query": original_query,
        "latest_user_message": context.latest_user_message,
        "conversation": list(context.conversation or []),
        "message_count": context.info.get("message_count", 0),
        "history_message_count": context.info.get("history_message_count", 0),
        "latest_user_index": context.info.get("latest_user_index"),
        "latest_user_length": context.info.get("latest_user_length", 0),
        "roles": list(context.info.get("roles") or []),
        "truncated": bool(context.info.get("truncated")),
    }


def _build_resolver_messages(
    *,
    context_history: list[ChatMessage],
    latest_user_message: str,
    structured_context: dict[str, Any],
) -> list[ChatMessage]:
    messages: list[ChatMessage] = [
        ChatMessage(role=ChatRole.SYSTEM, content=RESOLVER_SYSTEM_PROMPT)
    ]
    structured_context_json = json.dumps(
        structured_context,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    messages.append(
        ChatMessage(
            role=ChatRole.SYSTEM,
            content=f"Structured conversation context:\n{structured_context_json}",
        )
    )
    messages.extend(context_history)
    if latest_user_message:
        messages.append(ChatMessage(role=ChatRole.USER, content=latest_user_message))
    return messages


def _rewrite_latest_user_message(
    messages: list[ChatMessage],
    resolved_query: str,
) -> list[ChatMessage]:
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


def _conversation_resolution_metadata(
    *,
    original_query: str,
    resolved_query: str,
    resolution: ConversationResolution,
    applied: bool,
    elapsed_ms: float,
) -> dict[str, Any]:
    return {
        "original_query": original_query,
        "resolved_query": resolved_query,
        "model_resolved_query": resolution.resolved_query.strip(),
        "is_followup": applied and resolved_query != original_query,
        "model_is_followup": bool(resolution.is_followup),
        "confidence": float(resolution.confidence or 0.0),
        "applied": applied,
        "resolution_time_ms": elapsed_ms,
    }


def _log_resolution(
    *,
    original_query: str,
    resolved_query: str,
    resolution: ConversationResolution,
    elapsed_ms: float,
    applied: bool,
) -> None:
    entity_sources = {
        mention: entity.source for mention, entity in resolution.resolved_entities.items()
    }
    logger.debug(
        "conversation_resolver original_query=%r resolved_query=%r intent=%s "
        "conversation_subject=%r resolved_entities=%s entity_sources=%s "
        "entity_confidence=%.3f rewrite_confidence=%.3f resolution_time_ms=%.2f applied=%s",
        original_query,
        resolved_query,
        resolution.intent,
        resolution.conversation_subject,
        {
            mention: entity.model_dump(exclude_none=True)
            for mention, entity in resolution.resolved_entities.items()
        },
        entity_sources,
        float(resolution.entity_confidence or 0.0),
        float(resolution.rewrite_confidence or resolution.confidence or 0.0),
        elapsed_ms,
        applied,
    )


async def resolve_conversation_context(
    request: RequestState,
    *,
    settings: Settings,
    model_manager: ModelManager,
    ollama_client: OllamaClient,
) -> ResolverResult:
    original_query = str(request.original_query or request.user_message or "").strip()
    resolution = ConversationResolution(
        resolved_query=original_query,
        intent="NEW_TOPIC",
        conversation_subject="",
        resolved_entities={},
        entity_confidence=0.0,
        rewrite_confidence=0.0,
        confidence=0.0,
        is_followup=False,
    )

    context = build_resolver_context(request.messages)
    started = perf_counter()
    structured_context = _structured_context_payload(context, original_query=original_query)

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
                    "conversation_resolution": _conversation_resolution_metadata(
                        original_query=original_query,
                        resolved_query=original_query,
                        resolution=resolution,
                        applied=False,
                        elapsed_ms=elapsed_ms,
                    ),
                },
            }
        )
        _log_resolution(
            original_query=original_query,
            resolved_query=original_query,
            resolution=resolution,
            elapsed_ms=elapsed_ms,
            applied=False,
        )
        return ResolverResult(
            request=updated,
            resolution=resolution,
            resolution_time_ms=elapsed_ms,
            applied=False,
        )

    messages = _build_resolver_messages(
        context_history=context.history_messages,
        latest_user_message=context.latest_user_message or original_query,
        structured_context=structured_context,
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
        resolution = ConversationResolution(
            resolved_query=original_query,
            intent="NEW_TOPIC",
            conversation_subject="",
            resolved_entities={},
            entity_confidence=0.0,
            rewrite_confidence=0.0,
            confidence=0.0,
            is_followup=False,
        )

    applied = bool(
        resolution.confidence >= RESOLVER_CONFIDENCE_THRESHOLD
        and resolution.resolved_query.strip()
    )
    resolved_query = resolution.resolved_query.strip() if applied else original_query
    elapsed_ms = (perf_counter() - started) * 1000.0

    updated = request.model_copy(
        update={
            "original_query": original_query,
            "resolved_query": resolved_query,
            "user_message": resolved_query,
            "messages": _rewrite_latest_user_message(list(request.messages), resolved_query),
            "is_followup": applied and resolved_query != original_query,
            "followup_confidence": float(resolution.confidence or 0.0),
            "metadata": {
                **request.metadata,
                "conversation_resolution": _conversation_resolution_metadata(
                    original_query=original_query,
                    resolved_query=resolved_query,
                    resolution=resolution,
                    applied=applied,
                    elapsed_ms=elapsed_ms,
                ),
            },
        }
    )

    _log_resolution(
        original_query=original_query,
        resolved_query=resolved_query,
        resolution=resolution,
        elapsed_ms=elapsed_ms,
        applied=applied,
    )

    return ResolverResult(
        request=updated,
        resolution=resolution,
        resolution_time_ms=elapsed_ms,
        applied=applied,
    )
