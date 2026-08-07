"""Conversation assembly.

This module is the single authoritative layer for constructing outbound
conversations. It receives structured inputs only and never inspects raw
conversation history.

It guarantees, by construction, that every produced conversation satisfies
the generic OpenAI-compatible chat invariants shared by llama.cpp, Ollama,
OpenAI, vLLM, TGI, LM Studio and SGLang:

- at most one SYSTEM message
- a SYSTEM message, when present, is always first
- every conversation contains exactly one USER message carrying the actual
  request
- conversation history is passed through unchanged (role ordering preserved)
- roles are never encoded inside prompt text
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from ..common.enums import ChatRole
from ..logging import get_logger
from ..models.chat import ChatMessage

logger = get_logger(__name__)


def _pretty_json(content: str) -> str:
    """Pretty-print ``content`` when it is valid JSON, otherwise return as-is."""
    try:
        parsed = json.loads(content)
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except Exception:
        return content


def merge_system_sections(
    *,
    system_prompt: str,
    request_context: str = "",
    structured_context: str = "",
    additional_context: str = "",
) -> str:
    """Merge all system context into a single SYSTEM message string.

    Handles the case where callers historically supplied multiple context
    blobs (base prompt, request context, structured context, additional
    context, orchestration metadata). These are combined into one sectioned
    block so exactly one SYSTEM message is emitted.
    """
    system_sections: list[str] = [system_prompt.strip()]

    def append_section(title: str, content: str) -> None:
        content = content.strip()
        if not content:
            return
        system_sections.append(
            f"""## {title}

{_pretty_json(content)}"""
        )

    append_section("Request Context", request_context)
    append_section("Structured Context", structured_context)
    append_section("Additional Context", additional_context)

    return "\n\n".join(system_sections)


def _history_to_chat_messages(
    history: Iterable[ChatMessage] | None,
) -> list[ChatMessage]:
    return list(history or [])


def validate_conversation(messages: list[ChatMessage]) -> None:
    """Validate a produced conversation against the core invariants.

    This is a programming invariant, not a runtime guard. It fails loudly if
    any produced conversation would be rejected by an OpenAI-compatible chat
    backend.
    """
    system_count = 0
    for index, message in enumerate(messages):
        if message.role != ChatRole.SYSTEM:
            continue
        system_count += 1
        if index != 0:
            raise ValueError(
                "SYSTEM message must be the first message in the conversation"
            )

    if system_count > 1:
        raise ValueError(
            f"Conversation contains {system_count} SYSTEM messages; expected at most one"
        )

    has_user = any(message.role == ChatRole.USER for message in messages)
    if not has_user:
        raise ValueError("Conversation contains no user message")


def build_conversation(
    *,
    system_prompt: str,
    request_context: str = "",
    structured_context: str = "",
    additional_context: str = "",
    history: Iterable[ChatMessage] | None = None,
    latest_user_message: str | list[dict[str, Any]] | None = None,
) -> list[ChatMessage]:
    """Assemble a fully valid outbound conversation.

    Parameters are structured inputs only:

    - ``system_prompt``: the base system/role instructions.
    - ``request_context``: request metadata to merge into the SYSTEM message.
    - ``structured_context``: evidence/execution context to merge into SYSTEM.
    - ``additional_context``: any supplementary SYSTEM context.
    - ``history``: preserved conversation history (passed through unchanged).
    - ``latest_user_message``: the actual user request. This is mandatory.

    Returns a list of ``ChatMessage`` guaranteed to satisfy the invariants.
    """
    if not latest_user_message:
        raise ValueError(
            "build_conversation requires a latest_user_message; "
            "a conversation without a user message is invalid"
        )

    system_content = merge_system_sections(
        system_prompt=system_prompt,
        request_context=request_context,
        structured_context=structured_context,
        additional_context=additional_context,
    )

    messages: list[ChatMessage] = [ChatMessage(role=ChatRole.SYSTEM, content=system_content)]
    messages.extend(_history_to_chat_messages(history))
    messages.append(ChatMessage(role=ChatRole.USER, content=latest_user_message))

    validate_conversation(messages)

    return messages
