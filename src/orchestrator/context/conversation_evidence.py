"""Reusable conversation evidence policy (Feature 3).

This module owns the *policy* for persistent reusable specialist evidence:

- deterministic evidence identity
- conservative lookup / reuse decisions (vision, web, document)
- freshness override detection (explicit re-analysis / fresh search)
- lossless promotion of a reusable item back into the current
  ``EvidenceLedger`` (same structure a genuine specialist produces)
- persistence of successful specialist results into
  ``ConversationEvidenceState`` (bounded by ``Settings``)

The state model (``ConversationEvidenceState``) stays a simple container. All
specialist-specific matching and persistence policy lives here.

Persistence is owned by the existing LangGraph checkpoint. No database, cache
service, embeddings, or additional LLM call is introduced.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from ..common.enums import SpecialistType
from ..logging import get_logger
from ..models.evidence import (
    ConversationEvidenceItem,
    ConversationEvidenceState,
    EvidenceLedger,
    RepositoryEvidence,
    VisionEvidence,
    WebEvidence,
)
from ..models.state import ConversationState, OrchestratorState, RequestState
from ..settings import Settings

logger = get_logger(__name__)

# Evidence types persisted by Feature 3.
EVIDENCE_TYPE_VISION = "vision"
EVIDENCE_TYPE_WEB = "web"
EVIDENCE_TYPE_DOCUMENT = "document"

# Resolver intents that signal a clear topic/subject change.
_NEW_TOPIC_INTENT = "NEW_TOPIC"
_SUBJECT_SWITCH_INTENT = "SUBJECT_SWITCH"

# Explicit fresh/re-analysis phrases (deterministic keyword scan; no classifier).
_FRESH_ANALYSIS_PHRASES = (
    "analyze again",
    "analyse again",
    "re-analyze",
    "re-analyse",
    "reanalyze",
    "reanalyse",
    "look again",
    "re-read",
    "reread",
    "re-read the document",
    "process again",
    "search again",
    "search the web again",
    "fresh search",
    "fresh analysis",
    "fresh analyse",
    "ignore the previous analysis",
    "ignore the previous result",
    "new analysis",
    "do it again",
    "run again",
)

# Explicit freshness wording for web requests -> bypass stale web evidence.
_FRESHNESS_WEB_WORDS = (
    "latest",
    "current",
    "today",
    "right now",
    "now",
    "fresh",
    "up-to-date",
    "up to date",
    "newest",
)


def evidence_id(evidence_type: str, *parts: str) -> str:
    """Derive a short deterministic evidence identity.

    Uses the evidence type plus stable key parts (resource ids / normalized
    query) so identical sources map to the same id. Never hashes raw bytes.
    """
    key = ":".join([evidence_type, *(p for p in parts if p)]).strip(":")
    if not key:
        return ""
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return digest[:16]


def _normalize_text(text: str | None) -> str:
    return " ".join(str(text or "").split()).strip()


def _resolver_intent(request: RequestState) -> str:
    resolution = request.metadata.get("conversation_resolution") or {}
    if not isinstance(resolution, dict):
        return ""
    return str(resolution.get("intent") or "").strip().upper()


def _is_followup(request: RequestState) -> bool:
    if request.is_followup:
        return True
    return _resolver_intent(request) in {
        "FOLLOW_UP",
        "ELABORATION",
        "MODIFICATION",
        "RETRY",
        "COMPARISON",
        "CORRECTION",
        "CLARIFICATION",
    }


def _explicit_resource_reference(user_text: str, conversation: ConversationState) -> bool:
    """Heuristic: does the request explicitly refer back to an existing resource?

    Used to allow resource-specific evidence reuse even when the resolver
    labels the request as a topic transition. Conservative: only returns True
    on a clear mention of a known resource reference/name.
    """
    text = _normalize_text(user_text).lower()
    if not text:
        return False
    for resource in conversation.active_resources:
        ref = _normalize_text(resource.reference or resource.name or "").lower()
        if ref and ref in text:
            return True
    return False


def is_fresh_analysis_requested(request: RequestState) -> bool:
    """Detect explicit fresh/repeated processing requests.

    Inspects the normalized user message and the original query so neither an
    unspecified field nor a rewrite hides the explicit override. Deterministic
    keyword scan only; no LLM classifier.
    """
    user_text = _normalize_text(request.user_message or request.original_query or "").lower()
    if not user_text:
        return False
    for phrase in _FRESH_ANALYSIS_PHRASES:
        if phrase in user_text:
            return True
    return False


def _web_freshness_requested(request: RequestState) -> bool:
    """True when the request explicitly asks for current/latest web info."""
    text = _normalize_text(request.user_message or request.original_query or "").lower()
    for word in _FRESHNESS_WEB_WORDS:
        if word in text:
            return True
    return False


def _matches_topic(
    item: ConversationEvidenceItem,
    conversation: ConversationState,
    request: RequestState,
) -> bool:
    """Conservative topic/search-context match for web evidence.

    Does NOT require literal query substring overlap. Uses existing
    conversation signals: last web query, current topic, and (when present) the
    stored query. If no relationship can be established confidently, returns
    False -> treated as a cache miss.
    """
    if item.query and conversation.last_web_query:
        a = _normalize_text(item.query).lower()
        b = _normalize_text(conversation.last_web_query).lower()
        if a == b or (a and b and (a in b or b in a)):
            return True

    if conversation.current_topic:
        topic = _normalize_text(conversation.current_topic).lower()
        query = _normalize_text(item.query).lower()
        if query and topic and (query in topic or topic in query):
            return True

    # Fall back to the request's own resolved text when it clearly echoes the
    # stored query (handles "which of those affects networking?").
    request_text = _normalize_text(request.user_message or request.resolved_query or "").lower()
    if item.query and request_text:
        stored = _normalize_text(item.query).lower()
        if stored and stored in request_text:
            return True

    return False


def _validate_item(item: ConversationEvidenceItem, *, evidence_type: str) -> bool:
    """Minimum required-field validation before reuse.

    Malformed/unusable items are treated as a cache miss, never a hard failure.
    """
    if item is None:
        return False
    if item.evidence_type != evidence_type:
        return False
    if not _normalize_text(item.content):
        return False
    return True


# -- lookups -------------------------------------------------------------


def lookup_vision_evidence(
    state: OrchestratorState,
    settings: Settings,
) -> ConversationEvidenceItem | None:
    """Find reusable vision evidence for the active conversation resources.

    Does NOT require the current request to contain the image again: it uses
    ``ConversationState.active_resources``. A NEW unrelated topic blocks reuse
    unless the request explicitly references an existing resource.
    """
    conversation = state.conversation
    if not conversation.active_resources:
        return None

    if is_fresh_analysis_requested(state.request):
        logger.debug("specialist_evidence_miss specialist=vision reason=fresh_analysis_requested")
        return None

    intent = _resolver_intent(state.request)
    if intent in {_NEW_TOPIC_INTENT, _SUBJECT_SWITCH_INTENT}:
        if not _explicit_resource_reference(
            state.request.user_message or state.request.original_query or "",
            conversation,
        ):
            logger.debug("specialist_evidence_miss specialist=vision reason=new_topic_no_resource_reference")
            return None

    active_ids = {r.resource_id for r in conversation.active_resources if r.resource_id}

    for item in state.conversation_evidence.items:
        if item.evidence_type != EVIDENCE_TYPE_VISION:
            continue
        if not _validate_item(item, evidence_type=EVIDENCE_TYPE_VISION):
            continue
        # Reuse only when the evidence's resource is still active.
        if active_ids and item.resource_ids:
            if any(rid in active_ids for rid in item.resource_ids):
                logger.debug(
                    "specialist_evidence_hit specialist=vision evidence_id=%s resource_ids=%s",
                    item.evidence_id,
                    item.resource_ids,
                )
                return item

    logger.debug("specialist_evidence_miss specialist=vision")
    return None


def lookup_document_evidence(
    state: OrchestratorState,
    settings: Settings,
) -> ConversationEvidenceItem | None:
    """Find reusable document/file evidence for active file resources.

    Only applies when the conversation has an uploaded file/document resource.
    Arbitrary repository/RAG questions (no file resource) are NOT persistent
    document evidence and are never reused here.
    """
    conversation = state.conversation
    if not conversation.active_resources:
        return None

    if is_fresh_analysis_requested(state.request):
        logger.debug("specialist_evidence_miss specialist=document reason=fresh_analysis_requested")
        return None

    intent = _resolver_intent(state.request)
    if intent in {_NEW_TOPIC_INTENT, _SUBJECT_SWITCH_INTENT}:
        if not _explicit_resource_reference(
            state.request.user_message or state.request.original_query or "",
            conversation,
        ):
            logger.debug("specialist_evidence_miss specialist=document reason=new_topic_no_resource_reference")
            return None

    active_ids = {r.resource_id for r in conversation.active_resources if r.resource_id}

    for item in state.conversation_evidence.items:
        if item.evidence_type != EVIDENCE_TYPE_DOCUMENT:
            continue
        if not _validate_item(item, evidence_type=EVIDENCE_TYPE_DOCUMENT):
            continue
        if active_ids and item.resource_ids:
            if any(rid in active_ids for rid in item.resource_ids):
                logger.debug(
                    "specialist_evidence_hit specialist=document evidence_id=%s resource_ids=%s",
                    item.evidence_id,
                    item.resource_ids,
                )
                return item

    logger.debug("specialist_evidence_miss specialist=document")
    return None


def lookup_web_evidence(
    state: OrchestratorState,
    settings: Settings,
) -> ConversationEvidenceItem | None:
    """Find reusable web evidence for a follow-up to a previous search.

    Conservative deterministic hierarchy:
      1. Explicit fresh/current/latest request -> no reuse.
      2. Not a follow-up -> no automatic reuse.
      3. Follow-up + existing web evidence + matching topic/search context
         -> eligible for reuse (no literal substring overlap required).
      4. Otherwise -> cache miss (fresh search).
    ``has_web_results`` alone is never sufficient.
    """
    conversation = state.conversation
    if not conversation.has_web_results:
        return None

    if is_fresh_analysis_requested(state.request):
        logger.debug("specialist_evidence_miss specialist=web reason=fresh_analysis_requested")
        return None

    if _web_freshness_requested(state.request):
        logger.debug("specialist_evidence_miss specialist=web reason=freshness_requested")
        return None

    if not _is_followup(state.request):
        logger.debug("specialist_evidence_miss specialist=web reason=not_followup")
        return None

    for item in state.conversation_evidence.items:
        if item.evidence_type != EVIDENCE_TYPE_WEB:
            continue
        if not _validate_item(item, evidence_type=EVIDENCE_TYPE_WEB):
            continue
        if _matches_topic(item, conversation, state.request):
            logger.debug(
                "specialist_evidence_hit specialist=web evidence_id=%s query=%r",
                item.evidence_id,
                item.query[:120],
            )
            return item

    logger.debug("specialist_evidence_miss specialist=web")
    return None


# -- promotion -----------------------------------------------------------


def _deserialize_content(item: ConversationEvidenceItem) -> dict[str, Any] | None:
    try:
        parsed = json.loads(item.content or "")
        if isinstance(parsed, dict):
            return parsed
    except (ValueError, TypeError):
        logger.debug(
            "specialist_evidence_malformed evidence_id=%s evidence_type=%s",
            item.evidence_id,
            item.evidence_type,
        )
    return None


def promote_evidence(item: ConversationEvidenceItem, evidence: EvidenceLedger) -> EvidenceLedger:
    """Promote a reusable item into the current execution EvidenceLedger.

    Reconstructs the exact evidence structure a genuine specialist execution
    produces (VisionEvidence / WebEvidence / RepositoryEvidence). The
    validator/finalizer must see reused evidence exactly as fresh evidence.
    """
    if item is None:
        return evidence

    payload = _deserialize_content(item)
    if payload is None:
        logger.debug("specialist_evidence_promote_failed evidence_id=%s", getattr(item, "evidence_id", ""))
        return evidence

    try:
        if item.evidence_type == EVIDENCE_TYPE_VISION:
            evidence.vision = VisionEvidence.model_validate(payload)
        elif item.evidence_type == EVIDENCE_TYPE_WEB:
            evidence.web = WebEvidence.model_validate(payload)
        elif item.evidence_type == EVIDENCE_TYPE_DOCUMENT:
            evidence.repository = RepositoryEvidence.model_validate(payload)
        else:
            return evidence
    except Exception:
        logger.exception("specialist_evidence_promote_failed evidence_id=%s", item.evidence_id)
        return evidence

    logger.debug(
        "specialist_evidence_reused specialist=%s evidence_id=%s evidence_type=%s",
        item.specialist,
        item.evidence_id,
        item.evidence_type,
    )
    return evidence


# -- persistence ---------------------------------------------------------


def _serialize_content(evidence: BaseModel) -> str:
    return json.dumps(evidence.model_dump(exclude_none=True), ensure_ascii=False, separators=(",", ":"))


def _bounded_content(content: str, settings: Settings) -> str:
    """Apply the configured per-item content length limit.

    Truncates only when the truncated result still leaves meaningful evidence.
    Otherwise returns an empty string so the caller can skip persisting.
    """
    limit = max(1, settings.conversation_evidence_max_content_length)
    if len(content) <= limit:
        return content
    truncated = content[:limit]
    if len(_normalize_text(truncated)) < 8:
        return ""
    return truncated


def _prune(state: ConversationEvidenceState, settings: Settings) -> ConversationEvidenceState:
    """Enforce max item count and max total char bounds (newest-wins eviction)."""
    max_items = max(0, settings.conversation_evidence_max_items)
    max_total = max(0, settings.conversation_evidence_max_total_chars)

    items = list(state.items)

    # Drop oldest first until within the per-item count bound.
    if len(items) > max_items:
        items = items[-max_items:]

    # Drop oldest first until within the total-char bound.
    while items and sum(len(i.content or "") + len(i.query or "") for i in items) > max_total:
        items.pop(0)

    return state.model_copy(update={"items": items})


def _replace_or_append(
    state: ConversationEvidenceState,
    item: ConversationEvidenceItem,
    settings: Settings,
) -> ConversationEvidenceState:
    """Replace a matching reusable item (same identity) or append a new one.

    Fresh success supersedes prior evidence for the same resource/query
    identity; unrelated evidence is preserved.
    """
    kept = [existing for existing in state.items if existing.evidence_id != item.evidence_id]
    kept.append(item)
    return _prune(state.model_copy(update={"items": kept}), settings)


def persist_vision_evidence(
    state: OrchestratorState,
    evidence: EvidenceLedger,
    resource_ids: list[str],
    settings: Settings,
) -> OrchestratorState:
    """Persist a successful vision result as reusable evidence."""
    if evidence.vision is None:
        return state
    content = _bounded_content(_serialize_content(evidence.vision), settings)
    if not content:
        logger.debug("specialist_evidence_not_stored specialist=vision reason=content_limit")
        return state
    item = ConversationEvidenceItem(
        evidence_id=evidence_id(EVIDENCE_TYPE_VISION, *resource_ids),
        evidence_type=EVIDENCE_TYPE_VISION,
        specialist=SpecialistType.VISION.value,
        resource_ids=list(resource_ids),
        query="",
        content=content,
        turn_metadata={"image_count": len(resource_ids)},
        created_at=datetime.utcnow(),
        metadata={"status": "success", "reused": False},
    )
    state.conversation_evidence = _replace_or_append(
        state.conversation_evidence, item, settings
    )
    logger.debug(
        "specialist_evidence_stored specialist=vision evidence_id=%s resource_ids=%s",
        item.evidence_id,
        item.resource_ids,
    )
    return state


def persist_web_evidence(
    state: OrchestratorState,
    evidence: EvidenceLedger,
    query: str,
    settings: Settings,
) -> OrchestratorState:
    """Persist a successful web result as reusable evidence."""
    if evidence.web is None:
        return state
    content = _bounded_content(_serialize_content(evidence.web), settings)
    if not content:
        logger.debug("specialist_evidence_not_stored specialist=web reason=content_limit")
        return state
    normalized = _normalize_text(query or evidence.web.query or "")
    item = ConversationEvidenceItem(
        evidence_id=evidence_id(EVIDENCE_TYPE_WEB, normalized),
        evidence_type=EVIDENCE_TYPE_WEB,
        specialist=SpecialistType.WEB.value,
        resource_ids=[],
        query=normalized,
        content=content,
        turn_metadata={"query": normalized},
        created_at=datetime.utcnow(),
        metadata={"status": "success", "reused": False},
    )
    state.conversation_evidence = _replace_or_append(
        state.conversation_evidence, item, settings
    )
    logger.debug(
        "specialist_evidence_stored specialist=web evidence_id=%s query=%r",
        item.evidence_id,
        normalized[:120],
    )
    return state


def persist_document_evidence(
    state: OrchestratorState,
    evidence: EvidenceLedger,
    resource_ids: list[str],
    settings: Settings,
) -> OrchestratorState:
    """Persist a successful document/file processing result as reusable evidence.

    Maps back to the existing ``RepositoryEvidence`` representation.
    """
    if evidence.repository is None:
        return state
    content = _bounded_content(_serialize_content(evidence.repository), settings)
    if not content:
        logger.debug("specialist_evidence_not_stored specialist=document reason=content_limit")
        return state
    item = ConversationEvidenceItem(
        evidence_id=evidence_id(EVIDENCE_TYPE_DOCUMENT, *resource_ids),
        evidence_type=EVIDENCE_TYPE_DOCUMENT,
        specialist=SpecialistType.KNOWLEDGE.value,
        resource_ids=list(resource_ids),
        query=str(evidence.repository.question or ""),
        content=content,
        turn_metadata={},
        created_at=datetime.utcnow(),
        metadata={"status": "success", "reused": False},
    )
    state.conversation_evidence = _replace_or_append(
        state.conversation_evidence, item, settings
    )
    logger.debug(
        "specialist_evidence_stored specialist=document evidence_id=%s resource_ids=%s",
        item.evidence_id,
        item.resource_ids,
    )
    return state


# -- controller availability metadata ------------------------------------


def render_reusable_evidence_summary(state: OrchestratorState) -> str:
    """Render a compact, deterministic availability view for the planner.

    Only metadata (evidence type + key reference), NEVER full evidence bodies,
    keeping controller context bounded.
    """
    lines: list[str] = ["Reusable Specialist Evidence:"]
    if not state.conversation_evidence.items:
        lines.append("- none")
        return "\n".join(lines)

    for item in state.conversation_evidence.items:
        if item.evidence_type == EVIDENCE_TYPE_VISION:
            refs = ", ".join(item.resource_ids[:4]) or "?"
            lines.append(f"- VISION for resource(s): {refs}")
        elif item.evidence_type == EVIDENCE_TYPE_DOCUMENT:
            refs = ", ".join(item.resource_ids[:4]) or "?"
            lines.append(f"- DOCUMENT for resource(s): {refs}")
        elif item.evidence_type == EVIDENCE_TYPE_WEB:
            lines.append(f"- WEB for previous query: {item.query[:120] or '?'}")

    return "\n".join(lines)


__all__ = [
    "EVIDENCE_TYPE_VISION",
    "EVIDENCE_TYPE_WEB",
    "EVIDENCE_TYPE_DOCUMENT",
    "evidence_id",
    "is_fresh_analysis_requested",
    "lookup_vision_evidence",
    "lookup_document_evidence",
    "lookup_web_evidence",
    "promote_evidence",
    "persist_vision_evidence",
    "persist_web_evidence",
    "persist_document_evidence",
    "render_reusable_evidence_summary",
]