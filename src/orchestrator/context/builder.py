from __future__ import annotations

from typing import Any

from ..graph.prompts import (
    BASE_SYSTEM_PROMPT,
    CLARIFY_SYSTEM_PROMPT,
    CODE_SYSTEM_PROMPT,
    RAG_SYSTEM_PROMPT,
    TOOLS_SYSTEM_PROMPT,
    VISION_SYSTEM_PROMPT,
)
from ..schemas import ChatMessage, ChatRole, RouteDecision, RouteType
from ..settings import Settings
from ..vision.prompts import build_vision_injection_message
from .validator import RetrievalValidationResult


def last_user_text(messages: list[dict[str, Any]] | None) -> str:
    if not messages:
        return ""

    for message in reversed(messages):
        if message.get("role") == ChatRole.USER.value and message.get("content"):
            content = message["content"]
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = str(part.get("text", "")).strip()
                        if text:
                            text_parts.append(text)
                return "\n".join(text_parts).strip()
            return str(content)
    return ""


def resolve_system_prompt_for_route(decision: RouteDecision) -> str:
    if decision.route == RouteType.VISION:
        return "\n\n".join([BASE_SYSTEM_PROMPT, VISION_SYSTEM_PROMPT])
    if decision.route == RouteType.CODE:
        return "\n\n".join([BASE_SYSTEM_PROMPT, CODE_SYSTEM_PROMPT])
    if decision.route == RouteType.RAG:
        return "\n\n".join([BASE_SYSTEM_PROMPT, RAG_SYSTEM_PROMPT])
    if decision.route == RouteType.TOOLS:
        return "\n\n".join([BASE_SYSTEM_PROMPT, TOOLS_SYSTEM_PROMPT])
    if decision.route == RouteType.CLARIFY:
        return "\n\n".join([BASE_SYSTEM_PROMPT, CLARIFY_SYSTEM_PROMPT])
    if decision.route == RouteType.MULTI_STEP:
        return BASE_SYSTEM_PROMPT
    return BASE_SYSTEM_PROMPT


def select_model_for_route(settings: Settings, decision: RouteDecision) -> str:
    if decision.route == RouteType.VISION:
        return settings.vision_model
    if decision.route == RouteType.CODE:
        return settings.coder_model
    return settings.general_model


def _state_messages_to_chat_messages(messages: list[dict[str, Any]] | None) -> list[ChatMessage]:
    if not messages:
        return []
    return [ChatMessage.model_validate(message) for message in messages]


def _message_text_matches_last_user(messages: list[ChatMessage], latest_user_message: str) -> bool:
    if not messages or not latest_user_message:
        return False

    last_message = messages[-1]
    if last_message.role != ChatRole.USER:
        return False

    if isinstance(last_message.content, str):
        return last_message.content.strip() == latest_user_message.strip()

    return False


def _build_retrieval_metadata_message(validation: RetrievalValidationResult, intent: str) -> ChatMessage:
    return ChatMessage(
        role=ChatRole.SYSTEM,
        content=f"""
Knowledge Retrieval

Intent:
{intent}

Grounded:
{validation.grounded}

Hits:
{validation.hit_count}

Best Score:
{validation.score:.3f}
""".strip(),
    )


def build_generation_messages(
    *,
    system_prompt: str,
    messages: list[dict[str, Any]] | None = None,
    vision_context: str = "",
    knowledge_context: str = "",
    validation: RetrievalValidationResult | None = None,
    retrieval_metadata_intent: str = "unknown",
    mcp_context: str = "",
    memory_context: str = "",
    latest_user_message: str | None = None,
) -> list[ChatMessage]:
    outgoing: list[ChatMessage] = [ChatMessage(role=ChatRole.SYSTEM, content=system_prompt)]

    if vision_context:
        user_text = latest_user_message or last_user_text(messages)
        outgoing.append(
            ChatMessage(
                role=ChatRole.SYSTEM,
                content=build_vision_injection_message(vision_context, user_text),
            )
        )

    if validation is not None and validation.context:
        outgoing.append(ChatMessage(role=ChatRole.SYSTEM, content=validation.context))
    elif knowledge_context:
        outgoing.append(ChatMessage(role=ChatRole.SYSTEM, content=knowledge_context))

    if validation is not None:
        outgoing.append(_build_retrieval_metadata_message(validation, retrieval_metadata_intent))

    if mcp_context:
        outgoing.append(ChatMessage(role=ChatRole.SYSTEM, content=mcp_context))

    if memory_context:
        outgoing.append(ChatMessage(role=ChatRole.SYSTEM, content=memory_context))

    history_messages = _state_messages_to_chat_messages(messages)
    outgoing.extend(history_messages)

    if latest_user_message and not _message_text_matches_last_user(history_messages, latest_user_message):
        outgoing.append(ChatMessage(role=ChatRole.USER, content=latest_user_message))

    return outgoing
