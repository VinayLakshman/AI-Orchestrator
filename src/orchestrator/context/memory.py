"""Bounded, checkpoint-safe conversation memory.

This module deliberately stores only sanitized text and lightweight references.
It is the shared source for follow-up resolution and controller context; it is
not a second evidence store and never persists raw attachments or debug data.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ..common.enums import ChatRole
from ..models.chat import ChatMessage
from ..models.state import (
    ConversationMemoryState,
    ConversationTurn,
    OrchestratorState,
)
from ..settings import Settings
from .conversation_builder import ConversationContextBuilder, estimate_text_tokens


def _text(value: Any) -> str:
    if isinstance(value, str):
        return " ".join(value.split()).strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                for key in ("text", "content", "url"):
                    candidate = item.get(key)
                    if isinstance(candidate, str) and candidate.strip():
                        parts.append(candidate.strip())
                        break
        return "\n".join(parts).strip()
    return "" if value is None else str(value).strip()


def _truncate(value: str, limit: int) -> str:
    value = _text(value)
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)].rstrip() + "…"


_DATA_URL_RE = re.compile(
    r"data:[^,;]+(?:;[^,]*)?,[A-Za-z0-9+/=\s]+",
    re.IGNORECASE,
)


def _safe_content(content: Any) -> str:
    if isinstance(content, str):
        return _DATA_URL_RE.sub("[Attachment Attached]", content)
    if not isinstance(content, list):
        return _text(content)

    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "").lower()
        if item_type == "text":
            text = _text(item.get("text") or item.get("content"))
            if text:
                parts.append(_DATA_URL_RE.sub("[Attachment Attached]", text))
            continue
        if "image" in item_type or "file" in item_type or "document" in item_type:
            parts.append("[Attachment Attached]")
            continue
        value = item.get("text") or item.get("content")
        if isinstance(value, str) and value.strip():
            parts.append(_DATA_URL_RE.sub("[Attachment Attached]", value.strip()))
    return "\n".join(parts).strip()


def _safe_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
    result: list[ChatMessage] = []
    for message in messages:
        content = _truncate(_safe_content(message.content), 6000)
        result.append(message.model_copy(update={"content": content}))
    return result


def _latest_user(messages: list[ChatMessage]) -> ChatMessage | None:
    for message in reversed(messages):
        if message.role == ChatRole.USER:
            return message
    return None


def _has_client_history(messages: list[ChatMessage]) -> bool:
    user_count = sum(message.role == ChatRole.USER for message in messages)
    return user_count > 1 or any(
        message.role in {ChatRole.ASSISTANT, ChatRole.TOOL}
        for message in messages
    )


def _turn_text(turn: ConversationTurn) -> str:
    return " ".join(
        item
        for item in (turn.user_text, turn.assistant_text, turn.assistant_summary)
        if item
    )


def _terms(value: str) -> set[str]:
    return {part.lower() for part in _text(value).split() if len(part) > 2}


def _select_turns(
    state: OrchestratorState,
    settings: Settings,
) -> list[ConversationTurn]:
    memory = state.conversation.memory
    turns = list(memory.turns)
    if not turns:
        return []

    max_turns = max(0, settings.conversation_memory_max_turns)
    if max_turns == 0:
        return []
    recent_count = max(0, settings.conversation_memory_recent_turns)
    recent = turns[-min(recent_count, max_turns):] if recent_count else []
    recent_ids = {turn.turn_id for turn in recent}
    query_terms = _terms(state.request.original_query or state.request.user_message)
    topic_terms = _terms(state.conversation.current_topic)
    resource_terms = {
        resource.resource_id
        for resource in state.conversation.active_resources
        if resource.resource_id
    }

    scored: list[tuple[int, int, ConversationTurn]] = []
    for index, turn in enumerate(turns[:-recent_count] if recent_count else turns):
        score = 0
        turn_terms = _terms(_turn_text(turn))
        score += min(8, len(query_terms & turn_terms)) * 3
        score += min(4, len(topic_terms & turn_terms))
        if resource_terms & set(turn.resource_ids):
            score += 5
        if turn.artifacts:
            score += 1
        scored.append((score, index, turn))

    # Deterministic ranking: relevance first, then recency. The semantic
    # resolver still owns intent decisions; this only chooses context.
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = list(recent)
    for _, _, turn in scored:
        if turn.turn_id not in recent_ids:
            selected.append(turn)
        if len(selected) >= max_turns:
            break

    selected.sort(key=lambda turn: turn.created_at)
    return selected


@dataclass(slots=True)
class ConversationMemoryContext:
    history_messages: list[ChatMessage]
    structured: dict[str, Any]
    source: str
    selected_turns: int
    truncated: bool = False


def build_memory_context(
    state: OrchestratorState,
    settings: Settings,
    *,
    token_budget: int | None = None,
) -> ConversationMemoryContext:
    """Build one bounded context view for resolver/controller consumers."""
    memory = state.conversation.memory
    current = _safe_messages(list(state.request.messages))
    selected = _select_turns(state, settings) if settings.conversation_memory_enabled else []

    if _has_client_history(current):
        history = current
        source = "request_history"
        selected_count = 0
    else:
        historical: list[ChatMessage] = []
        for turn in selected:
            historical.extend(_safe_messages(turn.messages))
        history = historical + current
        source = "checkpoint_memory" if historical else "request_only"
        selected_count = len(selected)

    budget = token_budget or settings.conversation_memory_max_tokens
    builder = ConversationContextBuilder(token_budget=max(1, budget))
    # Keep the current request outside the historical trimming decision when
    # possible; the builder's newest-message safety rule retains it.
    trimmed, info = builder.build(history)

    structured = {
        "memory_source": source,
        "selected_turns": selected_count,
        "memory_truncated": bool(info.truncated),
        "current_topic": state.conversation.current_topic,
        "topic_confidence": state.conversation.topic_confidence,
        "pending_clarification": memory.pending_clarification,
        "constraints": memory.constraints[-8:],
        "corrections": memory.corrections[-8:],
        "last_answer_artifacts": memory.last_answer_artifacts[-8:],
        "inherited_instructions": [
            {
                "role": str(message.role.value),
                "content": _truncate(_text(message.content), 1200),
            }
            for message in [*memory.system_messages[-2:], *memory.developer_messages[-2:]]
            if _text(message.content)
        ],
        "active_resources": [
            {
                "resource_id": resource.resource_id,
                "resource_type": resource.resource_type,
                "name": resource.name,
                "reference": resource.reference,
            }
            for resource in state.conversation.active_resources[-12:]
        ],
    }
    return ConversationMemoryContext(
        history_messages=trimmed,
        structured=structured,
        source=source,
        selected_turns=selected_count,
        truncated=bool(info.truncated),
    )


def render_memory_context(state: OrchestratorState, settings: Settings, *, token_budget: int | None = None) -> str:
    context = build_memory_context(state, settings, token_budget=token_budget)
    return json.dumps(context.structured, ensure_ascii=False, separators=(",", ":"), default=str)


def _compact_turn(turn: ConversationTurn) -> ConversationTurn:
    summary = turn.assistant_summary or _truncate(turn.assistant_text, 900)
    messages = [
        message
        for message in turn.messages
        if message.role not in {ChatRole.ASSISTANT, ChatRole.TOOL}
    ]
    if summary:
        messages.append(ChatMessage(role=ChatRole.ASSISTANT, content=summary))
    return turn.model_copy(update={"assistant_text": summary, "messages": messages})


def _prune_memory(memory: ConversationMemoryState, settings: Settings) -> ConversationMemoryState:
    max_turns = max(0, settings.conversation_memory_max_turns)
    turns = list(memory.turns)[-max_turns:] if max_turns else []
    recent_count = max(0, settings.conversation_memory_recent_turns)
    compacted: list[ConversationTurn] = []
    for index, turn in enumerate(turns):
        compacted.append(
            turn if index >= max(0, len(turns) - recent_count) else _compact_turn(turn)
        )

    max_chars = max(0, settings.conversation_memory_max_chars)
    while compacted and sum(
        len(turn.user_text) + len(turn.assistant_text) + len(turn.assistant_summary)
        for turn in compacted
    ) > max_chars:
        compacted.pop(0)

    return memory.model_copy(update={"turns": compacted})


def commit_conversation_turn(state: OrchestratorState, settings: Settings) -> OrchestratorState:
    """Persist one completed response without retaining internal orchestration."""
    if not settings.conversation_memory_enabled:
        return state

    answer = _truncate(state.response.final_response, settings.conversation_memory_answer_max_chars)
    image_urls = state.response.metadata.get("image_urls")
    if not answer and state.response.metadata.get("route") == "image_generation":
        # Image URLs are persisted as artifacts; this placeholder gives the
        # resolver a conversational assistant turn without changing the
        # public image-generation response shape.
        answer = "Generated image artifact(s) are available in this conversation."
    if not answer:
        return state

    request_messages = _safe_messages(list(state.request.messages))
    user = _latest_user(request_messages)
    user_text = _text(user.content if user else state.request.original_query)
    resolution = state.request.metadata.get("conversation_resolution") or {}
    if not isinstance(resolution, dict):
        resolution = {}

    resource_ids = [
        resource.resource_id
        for resource in state.conversation.active_resources
        if resource.resource_id
    ][-12:]
    evidence_ids = [
        item.evidence_id
        for item in state.conversation_evidence.items[-6:]
        if item.evidence_id
    ]

    artifacts: list[dict[str, Any]] = []
    if isinstance(image_urls, str):
        image_urls = [image_urls]
    if isinstance(image_urls, list):
        artifacts.extend(
            {"type": "image", "url": str(url)[:2000]}
            for url in image_urls
            if isinstance(url, str) and url.strip()
        )
    if state.response.metadata.get("route"):
        artifacts.append({"type": "route", "value": str(state.response.metadata["route"])[:80]})

    assistant = ChatMessage(role=ChatRole.ASSISTANT, content=answer)
    turn_tool_messages: list[ChatMessage] = []
    if user is not None:
        user_index = next(
            (index for index, message in enumerate(request_messages) if message is user),
            len(request_messages),
        )
        turn_tool_messages = [
            message
            for message in request_messages[user_index + 1 :]
            if message.role == ChatRole.TOOL
        ][-4:]
    turn = ConversationTurn(
        turn_id=state.request.request_id or state.request.thread_id,
        request_id=state.request.request_id,
        messages=(
            ([ChatMessage(role=ChatRole.USER, content=user_text)] if user_text else [])
            + turn_tool_messages
            + [assistant]
        ),
        user_text=user_text,
        resolved_query=_truncate(state.request.resolved_query or user_text, 2000),
        assistant_text=answer,
        assistant_summary=_truncate(answer, 900),
        finish_reason=state.response.finish_reason,
        intent=str(resolution.get("intent") or ""),
        confidence=float(resolution.get("confidence") or state.request.followup_confidence or 0.0),
        resource_ids=resource_ids,
        evidence_ids=evidence_ids,
        artifacts=artifacts[:12],
        metadata={
            "is_followup": bool(state.request.is_followup),
            "reused_evidence": bool(state.execution.runtime.metadata.get("reused")),
            "specialists": [item.value for item in state.execution.runtime.completed[-8:]],
        },
    )

    current_system = [m for m in request_messages if m.role == ChatRole.SYSTEM]
    # ChatRole currently models the public roles accepted by this service. Keep
    # a future developer role harmless if the schema adds it later.
    current_developer = [m for m in request_messages if str(m.role.value) == "developer"]
    memory = state.conversation.memory.model_copy(
        update={
            "turns": [*state.conversation.memory.turns, turn],
            "system_messages": current_system[-4:] or state.conversation.memory.system_messages,
            "developer_messages": current_developer[-4:] or state.conversation.memory.developer_messages,
            "last_answer_artifacts": artifacts[:12],
            "pending_clarification": (
                {"question": answer, "turn_id": turn.turn_id}
                if state.response.finish_reason == "clarify"
                else {}
            ),
            "corrections": (
                [*state.conversation.memory.corrections, user_text][-8:]
                if str(resolution.get("intent") or "").upper() == "CORRECTION"
                else state.conversation.memory.corrections
            ),
        }
    )
    state.conversation = state.conversation.model_copy(
        update={"memory": _prune_memory(memory, settings)}
    )
    return state


__all__ = [
    "ConversationMemoryContext",
    "build_memory_context",
    "commit_conversation_turn",
    "render_memory_context",
]
