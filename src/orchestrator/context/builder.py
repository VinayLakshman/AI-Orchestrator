from __future__ import annotations

from typing import Any

from ..common.enums import ChatRole
from ..models.chat import ChatMessage
from ..models.knowledge import KnowledgeRetrieveResponse
from ..models.ollama import ModelGenerationResponse
from ..schemas import (
    ControllerPlan,
    ControllerValidation,
    CoderResult,
    NormalizedRequest,
    ToolResult,
)

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
            parts: list[str] = []

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


def _state_messages_to_chat_messages(
    messages: list[dict[str, Any]] | None,
) -> list[ChatMessage]:
    if not messages:
        return []

    return [ChatMessage.model_validate(message) for message in messages]


def render_structured_context(
    *,
    vision_context: str = "",
    knowledge_result: KnowledgeRetrieveResponse | None = None,
    coder_result: CoderResult | None = None,
    tool_result: ToolResult | None = None,
    reasoning_result: ModelGenerationResponse | None = None,
    controller_plan: ControllerPlan | None = None,
    controller_validation: ControllerValidation | None = None,
) -> str:
    parts: list[str] = []

    def add(title: str, value: str) -> None:
        value = (value or "").strip()
        if value:
            parts.extend([f"## {title}", value, ""])

    if controller_plan:
        add("Controller Plan", controller_plan.model_dump_json(indent=2))
    if controller_validation and (
        not controller_plan
        or controller_validation.model_dump(exclude_none=True)
        != controller_plan.model_dump(exclude_none=True)
    ):
        add("Controller Validation", controller_validation.model_dump_json(indent=2))

    if knowledge_result and knowledge_result.context:
        add("Knowledge Context", knowledge_result.context)

    if vision_context:
        add("Vision Context", vision_context)

    if coder_result and (coder_result.summary or coder_result.code):
        add(
            "Coder Result",
            coder_result.model_dump_json(indent=2),
        )

    if tool_result and (tool_result.summary or tool_result.result):
        add(
            "Tool Result",
            tool_result.model_dump_json(indent=2),
        )

    if reasoning_result and reasoning_result.content:
        add("Reasoning Result", reasoning_result.content)

    return "\n".join(parts).strip()


def render_request_context(request: NormalizedRequest | dict[str, Any] | None) -> str:
    if request is None:
        return ""
    if isinstance(request, dict):
        try:
            request = NormalizedRequest.model_validate(request)
        except Exception:
            return ""

    parts = [
        "# Normalized Request",
        f"- User query: {request.user_query or ''}",
        f"- Message count: {request.metadata.get('message_count', 0)}",
        f"- Has images: {bool(request.metadata.get('has_images', False))}",
        f"- Has files: {bool(request.metadata.get('has_files', False))}",
        f"- Attachment types: {', '.join(request.metadata.get('attachment_types', [])) or 'none'}",
        f"- Contains URLs: {bool(request.metadata.get('contains_urls', False))}",
        f"- Contains code blocks: {bool(request.metadata.get('contains_code_blocks', False))}",
        f"- Estimated prompt tokens: {request.metadata.get('estimated_prompt_tokens', 0)}",
        "",
        "## Routing Hints",
        f"- repository_likelihood: {request.routing_hints.repository_likelihood:.2f}",
        f"- code_likelihood: {request.routing_hints.code_likelihood:.2f}",
        f"- vision_likelihood: {request.routing_hints.vision_likelihood:.2f}",
        "",
    ]

    if request.attachments:
        parts.append("## Attachments")
        for attachment in request.attachments[:8]:
            parts.append(f"- {attachment.attachment_type}: {attachment.placeholder}")
        parts.append("")

    return "\n".join(parts).strip()


def build_controller_messages(
    *,
    system_prompt: str,
    messages: list[dict[str, Any]] | None = None,
    request_context: str = "",
    vision_context: str = "",
    knowledge_result: KnowledgeRetrieveResponse | None = None,
    coder_result: CoderResult | None = None,
    tool_result: ToolResult | None = None,
    reasoning_result: ModelGenerationResponse | None = None,
    controller_plan: ControllerPlan | None = None,
    controller_validation: ControllerValidation | None = None,
    latest_user_message: str | None = None,
) -> list[ChatMessage]:
    outgoing: list[ChatMessage] = [ChatMessage(role=ChatRole.SYSTEM, content=system_prompt)]

    if request_context:
        outgoing.append(
            ChatMessage(
                role=ChatRole.SYSTEM,
                metadata={"source": "normalized_request"},
                content=request_context,
            )
        )

    structured_context = render_structured_context(
        vision_context=vision_context,
        knowledge_result=knowledge_result,
        coder_result=coder_result,
        tool_result=tool_result,
        reasoning_result=reasoning_result,
        controller_plan=controller_plan,
        controller_validation=controller_validation,
    )
    if structured_context:
        outgoing.append(
            ChatMessage(
                role=ChatRole.SYSTEM,
                metadata={"source": "structured_context"},
                content=structured_context,
            )
        )

    outgoing.extend(_state_messages_to_chat_messages(messages))

    if latest_user_message and last_user_text(messages) != latest_user_message:
        outgoing.append(ChatMessage(role=ChatRole.USER, content=latest_user_message))

    return outgoing
