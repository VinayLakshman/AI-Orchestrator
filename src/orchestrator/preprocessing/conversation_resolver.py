from __future__ import annotations

import json
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from pydantic import BaseModel, Field

from ..common.enums import ChatRole
from ..context.assembler import build_conversation
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
    "SUBJECT_SWITCH",
}
RESOLVER_ENTITY_SOURCES = {
    "previous_user_message",
    "previous_assistant_response",
    "recent_conversation",
    "latest_user_message",
    "external_fallback",
}

RESOLVER_SYSTEM_PROMPT = """You are an Intent & Context Resolver.

Your job is to understand the latest user request in the context of the recent
conversation and rewrite it into a fully self-contained request while
preserving the user's intended objective.

The resolver prepares requests for the orchestrator. It does not answer the
user's question or perform any execution.

## Core principles

- Use the recent conversation as the primary source of conversational meaning.
- Preserve the user's existing conversational objective unless the latest
  request clearly starts a NEW_TOPIC.
- Resolve references from conversation context when their meaning is clear.
- Preserve explicit details, constraints, targets, comparisons, and requested
  actions from the latest message.
- Do not invent entities, requirements, facts, or objectives.
- If the latest request is already self-contained, preserve it essentially
  unchanged.
- When the conversation does not provide enough evidence to resolve a
  reference confidently, do not invent a resolution.

## Conversation context

Use recent conversation entities first when resolving references, followed by
previous assistant responses and previous user requests.

Resolve conversational references such as:

- it, this, that, these, those
- them, they, he, she
- here, there
- former, latter
- first, second
- previous, above
- again, same

Conversation context should be used to understand what the user means, not to
force unrelated prior context into a new request.

A request that is short or elliptical may still be a valid continuation of the
conversation.

For example:

- "What about Docker?"
- "How about the second one?"
- "Can you explain that?"
- "Do it again."
- "What about USA?"

Interpret these using the preceding conversation when the intended meaning is
clear.

## Intent classification

Choose the intent that best represents the relationship between the latest
request and the preceding conversation.

The supported intents are:

- NEW_TOPIC:
  The request is independent of the previous conversational objective or
  clearly starts a different task.

- FOLLOW_UP:
  The request continues the existing subject or objective and may depend on
  previous context.

- ELABORATION:
  The user wants additional detail, explanation, or expansion of the previous
  answer while preserving its subject.

- MODIFICATION:
  The user wants to modify, revise, or change a previously generated artifact
  or result.

- RETRY:
  The user wants the previous request performed again without materially
  changing its objective.

- COMPARISON:
  The user continues the existing subject while introducing another target
  for comparison.

- CORRECTION:
  The user is correcting, challenging, or replacing information from the
  previous assistant response.

- CLARIFICATION:
  The user wants clarification of the immediately preceding explanation.

- SUBJECT_SWITCH:
  The conversational objective remains the same, but the primary subject is
  replaced.

For SUBJECT_SWITCH, preserve the existing objective while replacing only the
primary subject when that is what the user is doing.

For example:

Previous: "Explain networking in Kubernetes."
Latest: "What about Docker?"

The latest request should preserve the explanatory objective while changing the
subject.

Do not classify based only on keywords. Use the meaning of the latest request
and its relationship to the recent conversation.

## Objective preservation

Preserve the conversational objective when the latest request clearly
continues it.

If the latest message changes the subject but keeps the same objective, use
SUBJECT_SWITCH.

If the latest message changes the objective or clearly starts an unrelated
task, use NEW_TOPIC.

Do not force previous context into a NEW_TOPIC request merely because the new
request mentions an entity that appeared earlier.

## Rewriting the request

`resolved_query` must be understandable without requiring the reader to see
the previous conversation.

When appropriate, resolve implicit references and incorporate the minimum
necessary context required to make the request self-contained.

Preserve:

- the user's requested action
- important constraints
- qualifiers
- comparison targets
- quantities
- requested formats
- relevant entities
- the conversational objective

Do not unnecessarily expand a request that is already self-contained.

Do not answer the request.

Do not use tools.

Do not perform searches or external research.

Do not explain your reasoning.

## Confidence and ambiguity

Use confidence values to represent actual resolution certainty.

`entity_confidence`:
Confidence that referenced entities were correctly resolved.

`rewrite_confidence`:
Confidence that `resolved_query` accurately represents the user's intended
request.

`confidence`:
Overall confidence in the complete resolution.

Use high confidence when the conversation provides clear evidence.

If multiple interpretations remain plausible, do not manufacture certainty.
Preserve ambiguity where necessary and lower the relevant confidence value.

Set `is_followup` to true when the latest request materially depends on
previous conversation context. Otherwise set it to false.

## Important behavioral constraints

- Conversation context can override assumptions about what the user means.
- Do not replace explicit information from the latest user message with older
  information.
- Do not treat a mention of an earlier entity as proof that the request is a
  follow-up.
- Do not infer a new objective unless the latest message supports it.
- Do not infer a subject switch unless the conversational objective remains
  substantially the same.
- Do not resolve an ambiguous reference solely from world knowledge when the
  conversation does not establish the intended referent.
- Preserve the existing meaning of the request even when rewriting it into a
  self-contained form.

## Output contract

Return JSON only.

Return exactly these top-level fields:

{
  "intent": "...",
  "conversation_subject": "...",
  "resolved_query": "...",
  "resolved_entities": {},
  "conversation_objective": "...",
  "entity_confidence": 0.0,
  "rewrite_confidence": 0.0,
  "confidence": 0.0,
  "is_followup": false
}

`resolved_entities` must be an object.

Each resolved entity must contain:

{
  "resolved_to": "...",
  "source": "...",
  "confidence": 0.0
}

Do not include markdown, explanations, reasoning, or any text outside the JSON
object.
"""

class ResolvedEntity(BaseModel):
    resolved_to: str = ""
    source: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ConversationResolution(BaseModel):
    resolved_query: str = ""
    intent: str = "NEW_TOPIC"
    conversation_subject: str = ""
    conversation_objective: str = ""
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
    fallback_source = "external_fallback"
    if isinstance(value, dict):
        resolved_to = str(
            value.get("resolved_to")
            or value.get("resolved")
            or value.get("target")
            or value.get("entity")
            or value.get("value")
            or ""
        ).strip()
        source = str(value.get("source") or fallback_source).strip()
        confidence = _parse_float(value.get("confidence"), default=0.0)
        if source and source not in RESOLVER_ENTITY_SOURCES:
            source = fallback_source
        return ResolvedEntity(
            resolved_to=resolved_to,
            source=source,
            confidence=confidence,
        )

    resolved_to = str(value or "").strip()
    return ResolvedEntity(resolved_to=resolved_to, source=fallback_source, confidence=0.0)


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
    conversation_objective = str(
        parsed.get("conversation_objective")
        or parsed.get("objective")
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
        conversation_objective=conversation_objective,
        resolved_entities=resolved_entities,
        entity_confidence=entity_confidence,
        rewrite_confidence=rewrite_confidence,
        confidence=confidence,
        is_followup=bool(parsed.get("is_followup")),
    )


def _candidate_resolved_query(
    resolution: ConversationResolution,
    *,
    original_query: str,
) -> str:
    if resolution.intent == "NEW_TOPIC":
        return original_query
    candidate = resolution.resolved_query.strip()
    return candidate or original_query


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
    # Delegate to the centralized assembler. All orchestration metadata is
    # merged into the SYSTEM message so exactly one SYSTEM message is emitted,
    # the latest user request is a real USER message, and history order is
    # preserved (compatible with llama.cpp, Qwen and other OpenAI-compatible
    # chat APIs).
    try:
        structured_context_json = json.dumps(
            structured_context,
            ensure_ascii=False,
            indent=2,
        )
    except Exception:
        structured_context_json = json.dumps(
            structured_context,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    return build_conversation(
        system_prompt=RESOLVER_SYSTEM_PROMPT,
        structured_context=structured_context_json,
        history=context_history,
        latest_user_message=latest_user_message,
    )


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
        "intent": resolution.intent,
        "conversation_subject": resolution.conversation_subject,
        "conversation_objective": resolution.conversation_objective,
        "is_followup": applied and resolved_query != original_query,
        "model_is_followup": bool(resolution.is_followup),
        "entity_confidence": float(resolution.entity_confidence or 0.0),
        "rewrite_confidence": float(resolution.rewrite_confidence or resolution.confidence or 0.0),
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
        "conversation_subject=%r conversation_objective=%r resolved_entities=%s "
        "entity_sources=%s entity_confidence=%.3f rewrite_confidence=%.3f "
        "resolution_time_ms=%.2f applied=%s",
        original_query,
        resolved_query,
        resolution.intent,
        resolution.conversation_subject,
        resolution.conversation_objective,
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
    client_registry: Any,
    model_lifecycle: Any = None,
) -> ResolverResult:
    original_query = str(request.original_query or request.user_message or "").strip()
    resolution = ConversationResolution(
        resolved_query=original_query,
        intent="NEW_TOPIC",
        conversation_subject="",
        conversation_objective="",
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
        if model_lifecycle is not None:
            await model_lifecycle.ensure_warm("controller")

        response = await client_registry.get("controller").chat(
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
            conversation_objective="",
            resolved_entities={},
            entity_confidence=0.0,
            rewrite_confidence=0.0,
            confidence=0.0,
            is_followup=False,
        )

    applied = bool(
        resolution.confidence >= RESOLVER_CONFIDENCE_THRESHOLD
        and _candidate_resolved_query(resolution, original_query=original_query).strip()
    )
    resolved_query = (
        _candidate_resolved_query(resolution, original_query=original_query)
        if applied
        else original_query
    )
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
