from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..common.enums import ChatRole
from ..context.conversation_builder import (
    ConversationContextBuilder,
    DEFAULT_HISTORY_TOKEN_BUDGET,
)
from ..logging import get_logger
from ..models.chat import ChatMessage

logger = get_logger(__name__)

# Resolver history is token-budget driven (not a fixed message count).
# Default is 32000 tokens (increased from 12000) to support larger document processing
RESOLVER_HISTORY_TOKEN_BUDGET = DEFAULT_HISTORY_TOKEN_BUDGET


def _normalize_text(text: str | None) -> str:
    return " ".join(str(text or "").split()).strip()


def _message_role(message: dict[str, Any] | ChatMessage) -> str:
    if isinstance(message, ChatMessage):
        return str(message.role.value)
    return str(message.get("role") or "").strip().lower()


def _message_content(message: dict[str, Any] | ChatMessage) -> Any:
    if isinstance(message, ChatMessage):
        return message.content
    return message.get("content")


def _message_name(message: dict[str, Any] | ChatMessage) -> str | None:
    if isinstance(message, ChatMessage):
        return message.name
    value = message.get("name")
    return str(value).strip() if isinstance(value, str) and value.strip() else None


def _message_tool_call_id(message: dict[str, Any] | ChatMessage) -> str | None:
    if isinstance(message, ChatMessage):
        return message.tool_call_id
    value = message.get("tool_call_id")
    return str(value).strip() if isinstance(value, str) and value.strip() else None


def _message_metadata(message: dict[str, Any] | ChatMessage) -> dict[str, Any]:
    if isinstance(message, ChatMessage):
        return dict(message.metadata or {})
    metadata = message.get("metadata")
    return dict(metadata or {}) if isinstance(metadata, dict) else {}


def _message_timestamp(message: dict[str, Any] | ChatMessage) -> str | None:
    metadata = _message_metadata(message)
    for key in ("timestamp", "received_at", "created_at", "time"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if isinstance(message, dict):
        for key in ("timestamp", "received_at", "created_at", "time"):
            value = message.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _message_content_text(message: dict[str, Any] | ChatMessage) -> str:
    content = _message_content(message)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            for key in ("text", "content", "url"):
                value = part.get(key)
                if isinstance(value, str) and value.strip():
                    parts.append(value.strip())
                    break
        return "\n".join(parts).strip()

    if content is None:
        return ""

    return str(content).strip()


def _structural_entry(
    message: dict[str, Any] | ChatMessage,
    *,
    index: int,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "role": _message_role(message),
        "content": _message_content(message),
        "index": index,
    }

    name = _message_name(message)
    if name is not None:
        entry["name"] = name

    tool_call_id = _message_tool_call_id(message)
    if tool_call_id is not None:
        entry["tool_call_id"] = tool_call_id

    timestamp = _message_timestamp(message)
    if timestamp is not None:
        entry["timestamp"] = timestamp

    metadata = _message_metadata(message)
    if metadata:
        entry["metadata"] = metadata

    return entry


@dataclass(slots=True)
class ResolverContext:
    history_messages: list[ChatMessage]
    latest_user_message: str
    conversation: list[dict[str, Any]]
    info: dict[str, Any]

    @property
    def has_history(self) -> bool:
        return bool(self.history_messages)


def build_resolver_context(
    messages: list[dict[str, Any] | ChatMessage] | None,
    *,
    token_budget: int = RESOLVER_HISTORY_TOKEN_BUDGET,
) -> ResolverContext:
    raw_messages = list(messages or [])
    if not raw_messages:
        return ResolverContext(
            history_messages=[],
            latest_user_message="",
            conversation=[],
            info={
                "message_count": 0,
                "history_message_count": 0,
                "latest_user_index": None,
                "latest_user_length": 0,
                "roles": [],
                "truncated": False,
            },
        )

    latest_user_index: int | None = None
    for index in range(len(raw_messages) - 1, -1, -1):
        if _message_role(raw_messages[index]) == ChatRole.USER.value:
            latest_user_index = index
            break

    if latest_user_index is None:
        return ResolverContext(
            history_messages=[],
            latest_user_message="",
            conversation=[],
            info={
                "message_count": len(raw_messages),
                "history_message_count": 0,
                "latest_user_index": None,
                "latest_user_length": 0,
                "roles": [],
                "truncated": False,
            },
        )

    latest_user_message = _message_content_text(raw_messages[latest_user_index])

    # History is everything before the latest user message. The resolver only
    # feeds USER/ASSISTANT exchanges into the conversation builder, matching
    # the historical resolver contract. Token-budget-driven trimming (newest
    # first, oldest discarded) is delegated to the single authoritative
    # ConversationContextBuilder.
    history_messages: list[dict[str, Any] | ChatMessage] = [
        message
        for index, message in enumerate(raw_messages[:latest_user_index])
        if _message_role(message) in {ChatRole.USER.value, ChatRole.ASSISTANT.value}
    ]

    # Record the original index of each candidate so structural entries keep
    # their original position metadata.
    original_indices: dict[int, int] = {
        id(message): index
        for index, message in enumerate(raw_messages[:latest_user_index])
    }

    builder = ConversationContextBuilder(token_budget=token_budget)
    history_chat_messages, builder_info = builder.build(history_messages)

    # The builder trims oldest-first and returns the newest set in
    # chronological order. Therefore the kept raw messages are the trailing
    # ``len(history_chat_messages)`` entries of ``history_messages``.
    kept_raw = (
        history_messages[-len(history_chat_messages):]
        if history_chat_messages
        else []
    )

    conversation: list[dict[str, Any]] = []
    for message in kept_raw:
        original_index = original_indices.get(id(message))
        if original_index is None:
            continue
        conversation.append(_structural_entry(message, index=original_index))

    total_context_messages = sum(
        1
        for message in raw_messages[:latest_user_index]
        if _message_role(message) in {ChatRole.USER.value, ChatRole.ASSISTANT.value}
    )

    info = {
        "message_count": len(raw_messages),
        "history_message_count": len(history_chat_messages),
        "latest_user_index": latest_user_index,
        "latest_user_length": len(latest_user_message),
        "roles": [_message_role(message) for message in history_chat_messages],
        "truncated": bool(builder_info.truncated)
        or total_context_messages > len(history_chat_messages),
    }

    logger.debug(
        "resolver_context_window %s",
        {
            "message_count": info["message_count"],
            "history_message_count": info["history_message_count"],
            "latest_user_index": info["latest_user_index"],
            "truncated": info["truncated"],
        },
    )

    return ResolverContext(
        history_messages=history_chat_messages,
        latest_user_message=_normalize_text(latest_user_message),
        conversation=conversation,
        info=info,
    )
