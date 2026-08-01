from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..common.enums import ChatRole
from ..logging import get_logger
from ..models.chat import ChatMessage

logger = get_logger(__name__)

RESOLVER_HISTORY_MESSAGE_LIMIT = 6


def _normalize_text(text: str | None) -> str:
    return " ".join(str(text or "").split()).strip()


def _message_role(message: dict[str, Any] | ChatMessage) -> str:
    if isinstance(message, ChatMessage):
        return str(message.role.value)
    return str(message.get("role") or "").strip().lower()


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


@dataclass(slots=True)
class ResolverContext:
    history_messages: list[ChatMessage]
    latest_user_message: str
    info: dict[str, Any]

    @property
    def has_history(self) -> bool:
        return bool(self.history_messages)


def build_resolver_context(
    messages: list[dict[str, Any] | ChatMessage] | None,
    *,
    history_message_limit: int = RESOLVER_HISTORY_MESSAGE_LIMIT,
) -> ResolverContext:
    raw_messages = list(messages or [])
    if not raw_messages:
        return ResolverContext(history_messages=[], latest_user_message="", info={"message_count": 0})

    latest_user_index: int | None = None
    for index in range(len(raw_messages) - 1, -1, -1):
        role = _message_role(raw_messages[index])
        if role == ChatRole.USER.value:
            latest_user_index = index
            break

    if latest_user_index is None:
        return ResolverContext(history_messages=[], latest_user_message="", info={"message_count": len(raw_messages)})

    latest_user_message = _message_content_text(raw_messages[latest_user_index])

    history_messages: list[dict[str, Any] | ChatMessage] = []
    for message in reversed(raw_messages[:latest_user_index]):
        role = _message_role(message)
        if role not in {ChatRole.USER.value, ChatRole.ASSISTANT.value}:
            continue
        history_messages.append(message)
        if len(history_messages) >= history_message_limit:
            break

    history_messages.reverse()

    history_chat_messages = [ChatMessage.model_validate(message) for message in history_messages]
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
        "roles": [_message_role(message) for message in history_messages],
        "truncated": total_context_messages > len(history_chat_messages),
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
        info=info,
    )
