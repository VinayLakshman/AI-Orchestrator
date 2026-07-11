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


def last_user_text(messages: list[dict[str, Any]] | None) -> str:
    if not messages:
        return ""

    for message in reversed(messages):
        if message.get("role") != ChatRole.USER.value:
            continue

        content = message.get("content")

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts = []

            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "text"
                    and part.get("text")
                ):
                    parts.append(str(part["text"]).strip())

            return "\n".join(parts).strip()

        if content is not None:
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

    return BASE_SYSTEM_PROMPT


def select_model_for_route(
    settings: Settings,
    decision: RouteDecision,
) -> str:

    if decision.route == RouteType.VISION:
        return settings.vision_model

    if decision.route == RouteType.CODE:
        return settings.coder_model

    return settings.general_model


def _state_messages_to_chat_messages(
    messages: list[dict[str, Any]] | None,
) -> list[ChatMessage]:

    if not messages:
        return []

    return [
        ChatMessage.model_validate(message)
        for message in messages
    ]


def _build_retrieval_metadata_message(
    knowledge_result: dict[str, Any],
) -> ChatMessage:
    confidence = knowledge_result.get("confidence")
    confidence_text = (
        f"{float(confidence):.3f}"
        if confidence is not None
        else "n/a"
    )

    grounded = knowledge_result.get("grounded", False)
    intent = knowledge_result.get("intent") or "unknown"
    reason = knowledge_result.get("retrieval_reason") or "none"

    primary_hits = len(knowledge_result.get("primary_hits") or [])
    expanded_hits = len(knowledge_result.get("expanded_hits") or [])

    return ChatMessage(
        role=ChatRole.SYSTEM,
        metadata={
            "source": "knowledge_service",
            "type": "retrieval_metadata",
        },
        content=f"""
            Knowledge Retrieval Metadata

            Grounded: {grounded}
            Confidence: {confidence_text}
            Intent: {intent}
            Reason: {reason}

            Primary Hits: {primary_hits}
            Expanded Hits: {expanded_hits}

            The retrieved context that follows is the authoritative knowledge source for this request.
        """.strip(),
    )


def build_generation_messages(
    *,
    system_prompt: str,
    messages: list[dict[str, Any]] | None = None,
    vision_context: str = "",
    knowledge_result: dict[str, Any] | None = None,
    mcp_context: str = "",
    memory_context: str = "",
    latest_user_message: str | None = None,
) -> list[ChatMessage]:

    outgoing: list[ChatMessage] = [
        ChatMessage(
            role=ChatRole.SYSTEM,
            content=system_prompt,
        )
    ]

    #
    # Knowledge Service (highest priority context)
    #

    if knowledge_result:

        outgoing.append(
            _build_retrieval_metadata_message(
                knowledge_result
            )
        )

        context = (
            knowledge_result.get("context")
            or ""
        ).strip()

        if context:

            outgoing.append(
                ChatMessage(
                    role=ChatRole.SYSTEM,
                    metadata={
                        "source": "knowledge_service",
                        "grounded": knowledge_result.get("grounded", False),
                        "confidence": knowledge_result.get("confidence"),
                    },
                    content=f"""
                        The following information was retrieved from the user's indexed local knowledge base.

                        Treat this as the ONLY authoritative source for answering the user's question.

                        Do NOT use pretrained knowledge to fill gaps.
                        If the answer is not contained in this context, explicitly state that the documentation does not contain that information.

                        ========== BEGIN RETRIEVED CONTEXT ==========

                        {context}

                        =========== END RETRIEVED CONTEXT ===========
                    """.strip(),
                )
            )

    #
    # Vision Context
    #

    if vision_context:

        outgoing.append(
            ChatMessage(
                role=ChatRole.SYSTEM,
                metadata={
                    "source": "vision",
                },
                content=build_vision_injection_message(
                    vision_context,
                    latest_user_message
                    or last_user_text(messages),
                ),
            )
        )

    #
    # Memory Context
    #

    if memory_context:

        outgoing.append(
            ChatMessage(
                role=ChatRole.SYSTEM,
                metadata={
                    "source": "memory",
                },
                content=memory_context,
            )
        )

    #
    # MCP / Tool Context
    #

    if mcp_context:

        outgoing.append(
            ChatMessage(
                role=ChatRole.SYSTEM,
                metadata={
                    "source": "mcp",
                },
                content=mcp_context,
            )
        )

    #
    # Conversation History
    #

    history_messages = _state_messages_to_chat_messages(messages)
    outgoing.extend(history_messages)

    #
    # Fallback Latest User Message
    #

    if (
        latest_user_message
        and (
            not history_messages
            or last_user_text(messages) != latest_user_message
        )
    ):
        outgoing.append(
            ChatMessage(
                role=ChatRole.USER,
                content=latest_user_message,
            )
        )

    return outgoing