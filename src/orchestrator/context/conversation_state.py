"""ConversationState rendering and lightweight updates.

This module is the single authoritative place that manages the compact
``ConversationState`` layer. It intentionally owns ONLY the conversation-level
metadata responsibility:

- rendering a compact, deterministic text view for the controller
- extracting lightweight resource references from a request
- merging request resources into conversation resources (stable de-dup)
- conservative topic updates (establish / preserve / clear topic change)
- recording last successful specialist and web-search metadata

It deliberately does NOT:

- store resource contents, extracted text, OCR, embeddings or web result payloads
- build a second evidence cache (resources/web here are references/metadata only)
- add any LLM call, database, cache, or new persistence mechanism

Persistence is owned by the existing LangGraph checkpoint system.
"""

from __future__ import annotations

import hashlib

from typing import Any

from ..common.enums import SpecialistType
from ..logging import get_logger
from ..models.state import (
    ConversationResource,
    ConversationState,
    RequestState,
)

logger = get_logger(__name__)


# Resolver intents that signal a clear topic/subject change. Present in
# ``RequestState.metadata["conversation_resolution.intent"]`` when populated.
_NEW_TOPIC_INTENT = "NEW_TOPIC"
_SUBJECT_SWITCH_INTENT = "SUBJECT_SWITCH"


def _stable_id(reference: str, resource_type: str) -> str:
    """Derive a short deterministic resource identity.

    Uses the normalized reference so identical resources map to the same id.
    Never hashes/reads the binary asset itself.
    """
    key = f"{resource_type.lower()}:{reference}".strip(":")
    if not key:
        return ""
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return digest[:16]


def _reference_from_name(name: str) -> str:
    return name.strip()


# -- resource extraction ------------------------------------------------


def extract_resources_from_request(
    request: RequestState,
) -> list[ConversationResource]:
    """Derive lightweight resource references from a normalized request.

    Uses the existing normalized attachment info in
    ``RequestState.metadata["attachments"]`` and ``RequestState.images``.

    Only identity/reference metadata is stored; never the contents.
    """
    resources: list[ConversationResource] = []

    attachments: Any = request.metadata.get("attachments") or []
    if isinstance(attachments, list):
        for item in attachments:
            if not isinstance(item, dict):
                continue
            resource_type = str(item.get("attachment_type") or "").strip() or "file"
            reference = str(item.get("placeholder") or "").strip()
            name = str(item.get("reference") or "").strip()

            if not reference:
                raw = item.get("raw")
                name = name or _reference_from_name(str(
                    (raw.get("filename") if isinstance(raw, dict) else "") or ""
                ))
                reference = name or resource_type

            if not reference:
                continue

            resources.append(
                ConversationResource(
                    resource_id=_stable_id(reference, resource_type),
                    resource_type=resource_type,
                    reference=reference,
                    name=name,
                    metadata={"source": "request"},
                )
            )

    # Images are also recorded directly from RequestState.images (these are the
    # image placeholders/references). Skip any already captured via attachments.
    known_ids = {resource.resource_id for resource in resources}
    for reference in request.images or []:
        if not isinstance(reference, str) or not reference.strip():
            continue
        resource_id = _stable_id(reference, "image")
        if resource_id in known_ids:
            continue
        resources.append(
            ConversationResource(
                resource_id=resource_id,
                resource_type="image",
                reference=reference.strip(),
                name="",
                metadata={"source": "images"},
            )
        )

    return resources


def merge_request_resources(
    conversation: ConversationState,
    request: RequestState,
) -> ConversationState:
    """Merge current-request resource references into conversation resources.

    Stable de-dup by ``resource_id``: the same resource appearing in multiple
    requests is recorded once, never blindly appended.

    Resources persist across requests; removal/expiration is intentionally NOT
    implemented (a later feature may address resource lifecycle).
    """
    incoming = extract_resources_from_request(request)
    if not incoming:
        return conversation

    existing_by_id = {resource.resource_id: resource for resource in conversation.active_resources}

    added = 0
    merged = list(conversation.active_resources)
    for resource in incoming:
        if not resource.resource_id or resource.resource_id in existing_by_id:
            continue
        existing_by_id[resource.resource_id] = resource
        merged.append(resource)
        added += 1

    if added == 0:
        return conversation

    return conversation.model_copy(
        update={
            "active_resources": merged,
            "metadata": {
                **conversation.metadata,
                "resources_merged": int(conversation.metadata.get("resources_merged", 0) or 0) + added,
            },
        }
    )


# -- topic update -------------------------------------------------------


def _resolver_intent(request: RequestState) -> str:
    resolution = request.metadata.get("conversation_resolution") or {}
    if not isinstance(resolution, dict):
        return ""
    return str(resolution.get("intent") or "").strip().upper()


def update_topic(conversation: ConversationState, request: RequestState) -> ConversationState:
    """Conservatively update the current topic.

    Behavior:

    - empty topic  -> establish from the current request
    - follow-up    -> preserve the existing topic (do NOT blindly overwrite)
    - clear change -> replace the topic

    A clear change is recognized from the resolver intent (NEW_TOPIC /
    SUBJECT_SWITCH) when the resolver populated it meaningfully. Otherwise the
    topic is preserved and never faked with confidence values.
    """
    user_text = (request.user_message or request.resolved_query or "").strip()

    if not conversation.current_topic:
        if not user_text:
            return conversation
        return conversation.model_copy(
            update={
                "current_topic": user_text,
                "topic_confidence": 0.0,
            }
        )

    intent = _resolver_intent(request)

    needs_change = intent in {_NEW_TOPIC_INTENT, _SUBJECT_SWITCH_INTENT} and bool(user_text)
    if needs_change:
        return conversation.model_copy(
            update={
                "current_topic": user_text,
                "topic_confidence": 0.0,
                "metadata": {
                    **conversation.metadata,
                    "topic_updated": True,
                },
            }
        )

    # Follow-up / unknown intent: preserve the established topic.
    logger.debug(
        "conversation_state_preserved reason=%s thread_id=%s",
        "follow_up" if intent else "unknown_intent",
        request.thread_id,
    )
    return conversation


# -- specialist / web recording ----------------------------------------


def record_specialist_success(
    conversation: ConversationState,
    specialist: SpecialistType,
) -> ConversationState:
    """Record the last specialist that completed successfully."""
    existing = conversation.last_specialist
    updated = conversation.model_copy(
        update={
            "last_specialist": specialist,
        }
    )
    logger.debug(
        "conversation_state_update last_specialist=%s thread_id=%s",
        specialist.value,
        updated.metadata.get("thread_id", ""),
    )
    return updated


def record_web_success(
    conversation: ConversationState,
    *,
    query: str,
) -> ConversationState:
    """Record successful web-search metadata.

    Only call this on successful web execution. A failed search must not claim
    usable web results.
    """
    updated = conversation.model_copy(
        update={
            "has_web_results": True,
            "last_web_query": query.strip(),
        }
    )
    logger.debug(
        "conversation_state_update has_web_results=%s last_web_query=%r thread_id=%s",
        True,
        query.strip()[:200],
        updated.metadata.get("thread_id", ""),
    )
    return updated


def record_thread(conversation: ConversationState, thread_id: str) -> ConversationState:
    """Attach the thread id to conversation metadata (for logging only)."""
    if not thread_id:
        return conversation
    return conversation.model_copy(
        update={"metadata": {**conversation.metadata, "thread_id": thread_id}}
    )


# -- rendering ---------------------------------------------------------


def render_conversation_state(conversation: ConversationState) -> str:
    """Render a compact, deterministic text view of ConversationState.

    Empty/absent fields are omitted (no noisy placeholders).
    """
    lines: list[str] = ["Conversation State:"]

    if conversation.current_topic:
        lines.append(f"- Current topic: {conversation.current_topic}")

    if conversation.last_specialist is not None:
        lines.append(f"- Last specialist: {conversation.last_specialist.value}")

    if conversation.active_resources:
        rendered = []
        for resource in conversation.active_resources:
            resource_type = resource.resource_type or "resource"
            ref = resource.name or resource.reference or resource.resource_id
            rendered.append(f"{resource_type}:{ref}")
        lines.append("- Active resources: " + ", ".join(rendered[:12]))

    if conversation.has_web_results:
        lines.append("- Web results available: yes")

    if conversation.last_web_query:
        lines.append(f"- Last web query: {conversation.last_web_query}")

    return "\n".join(lines)

