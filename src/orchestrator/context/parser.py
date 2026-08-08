"""Conversation parsing and history management.

This module owns the *reading* side of conversation handling:

- extracting the latest user message
- preserving (and trimming) conversation history
- token budgeting
- deriving structural metadata

It intentionally knows nothing about constructing new conversations.
Construction is the responsibility of ``orchestrator.context.assembler``.
"""

from __future__ import annotations

import re
from typing import Any

from ..common.enums import ChatRole
from ..models.chat import ChatMessage

CONVERSATION_HISTORY_TOKEN_BUDGET = 2400


def estimate_text_tokens(text: str | None) -> int:
    """Estimate the number of tokens in ``text`` using a simple word count."""
    value = (text or "").strip()
    if not value:
        return 0
    return max(1, len(re.findall(r"\S+", value)))


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


def _message_sequence(
    messages: list[dict[str, Any] | ChatMessage],
) -> list[str]:
    return [_message_role(message) for message in messages]


def truncate_history(
    messages: list[ChatMessage],
    *,
    token_budget: int = CONVERSATION_HISTORY_TOKEN_BUDGET,
) -> tuple[list[ChatMessage], bool]:
    """Trim ``messages`` from the front until it fits within ``token_budget``.

    History is dropped oldest-first. The returned tuple is
    ``(kept_messages, truncated)``.
    """
    kept = list(messages)
    token_count = sum(_message_token_count(message) for message in kept)
    truncated = False

    while kept and token_count > token_budget:
        kept.pop(0)
        token_count = sum(_message_token_count(message) for message in kept)
        truncated = True

    return kept, truncated


def split_conversation(
    messages: list[dict[str, Any] | ChatMessage] | None,
    *,
    token_budget: int = CONVERSATION_HISTORY_TOKEN_BUDGET,
) -> tuple[list[ChatMessage], str, dict[str, Any]]:
    """Split a conversation into history, the latest user message, and metadata.

    Returns ``(history_chat_messages, latest_user_text, conversation_info)``.

    Raises ``ValueError`` if the conversation is empty or contains no user
    message. This intentionally enforces the invariant that an outbound
    conversation must always have a user message.
    """
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

    # History is everything before the latest user message. SYSTEM messages are
    # intentionally excluded: the assembler emits its own SYSTEM message, so
    # carrying a prior SYSTEM into history would produce a duplicated SYSTEM
    # (rejected by Qwen/llama.cpp). Other roles (USER, ASSISTANT, tool,
    # function, developer) are preserved verbatim to keep ordering intact.
    history_messages = [
        message
        for message in raw_messages[:latest_user_index]
        if _message_role(message) != ChatRole.SYSTEM.value
    ]

    history_token_count = sum(
        _message_token_count(message) for message in history_messages
    )

    truncated = False
    while history_messages and history_token_count > token_budget:
        history_messages.pop(0)
        history_token_count = sum(
            _message_token_count(message) for message in history_messages
        )
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
