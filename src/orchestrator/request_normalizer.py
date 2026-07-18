from __future__ import annotations

import copy
import math
import re
from typing import Any

from .common.enums import ChatRole
from .models.chat import ChatMessage
from .models.state import RequestState
from .schemas import (
    NormalizedAttachment,
    OpenAIChatCompletionRequest,
    OpenAIMessage,
)

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_CODE_BLOCK_RE = re.compile(r"```", re.DOTALL)


def _is_image_part(part: dict[str, Any]) -> bool:
    part_type = str(part.get("type") or "").lower()
    if part_type in {"image_url", "image", "input_image"}:
        return True
    mime_type = ""
    if isinstance(part.get("image_url"), dict):
        mime_type = str(part["image_url"].get("url") or "")
    if isinstance(part.get("mime_type"), str):
        mime_type = part["mime_type"]
    return "image" in part_type or mime_type.startswith("data:image/")


def _is_file_part(part: dict[str, Any]) -> bool:
    part_type = str(part.get("type") or "").lower()
    if part_type in {"file", "input_file", "document", "attachment", "pdf"}:
        return True
    filename = str(part.get("filename") or part.get("name") or part.get("path") or "").lower()
    mime_type = str(part.get("mime_type") or part.get("media_type") or "").lower()
    return filename.endswith((".pdf", ".txt", ".md", ".doc", ".docx", ".csv", ".json", ".yaml", ".yml")) or mime_type.startswith(
        "application/"
    )


def _attachment_type(part: dict[str, Any]) -> str:
    if _is_image_part(part):
        return "image"
    if _is_file_part(part):
        filename = str(part.get("filename") or part.get("name") or part.get("path") or "").lower()
        mime_type = str(part.get("mime_type") or part.get("media_type") or "").lower()
        if filename.endswith(".pdf") or "pdf" in mime_type:
            return "pdf"
        if filename.endswith((".md", ".txt", ".csv", ".json", ".yaml", ".yml", ".doc", ".docx")):
            return "document"
        return "file"
    return str(part.get("type") or "attachment").lower()


def _placeholder_for_attachment(attachment_type: str) -> str:
    return {
        "image": "<Image Attached>",
        "pdf": "<PDF Attached>",
        "document": "<Document Attached>",
        "file": "<File Attached>",
    }.get(attachment_type, "<Attachment Attached>")


def _attachment_reference(part: dict[str, Any], attachment_type: str) -> str:
    image_url = part.get("image_url")
    if isinstance(image_url, dict) and image_url.get("url"):
        return str(image_url["url"])
    for key in ("url", "source", "path", "file_id", "filename", "name"):
        value = part.get(key)
        if value:
            return str(value)
    return _placeholder_for_attachment(attachment_type)


def _content_to_text(content: Any, *, attachments: list[NormalizedAttachment]) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "").lower()
            if item_type == "text" and item.get("text") is not None:
                text = str(item.get("text") or "").strip()
                if text:
                    parts.append(text)
                continue
            if _is_image_part(item) or _is_file_part(item):
                attachment_type = _attachment_type(item)
                attachments.append(
                    NormalizedAttachment(
                        attachment_type=attachment_type,
                        placeholder=_attachment_reference(item, attachment_type),
                        raw=copy.deepcopy(item),
                    )
                )
                parts.append(_placeholder_for_attachment(attachment_type))
                continue
            value = item.get("text") or item.get("content") or item.get("url")
            if value:
                parts.append(str(value).strip())
        return "\n".join(part for part in parts if part).strip()
    if content is None:
        return ""
    return str(content).strip()


def _extract_controller_messages(
    messages: list[OpenAIMessage],
) -> tuple[list[dict[str, Any]], list[NormalizedAttachment], str]:
    controller_messages: list[dict[str, Any]] = []
    attachments: list[NormalizedAttachment] = []
    user_query = ""

    for message in messages:
        content = _content_to_text(message.content, attachments=attachments)
        controller_messages.append(
            ChatMessage(
                role=ChatRole(message.role),
                content=content,
                name=message.name,
                tool_call_id=message.tool_call_id,
            ).model_dump(exclude_none=True)
        )
        if message.role == ChatRole.USER.value and content and not user_query:
            user_query = content

    if not user_query:
        for message in reversed(controller_messages):
            if message.get("role") == ChatRole.USER.value and str(message.get("content") or "").strip():
                user_query = str(message.get("content") or "").strip()
                break

    return controller_messages, attachments, user_query


def _scan_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    for key in ("text", "content", "url"):
                        value = item.get(key)
                        if isinstance(value, str):
                            parts.append(value)
    return "\n".join(parts)


def normalize_openai_request(
    payload: OpenAIChatCompletionRequest,
    *,
    request_id: str = "",
    thread_id: str = "",
) -> RequestState:
    original_messages = [message.model_dump(exclude_none=True) for message in payload.messages]
    controller_messages, attachments, user_query = _extract_controller_messages(payload.messages)
    controller_text = _scan_text(controller_messages)

    has_images = any(item.attachment_type == "image" for item in attachments)
    has_files = any(item.attachment_type != "image" for item in attachments)
    attachment_types = list(dict.fromkeys(item.attachment_type for item in attachments))
    contains_urls = bool(_URL_RE.search(controller_text))
    contains_code_blocks = bool(_CODE_BLOCK_RE.search(controller_text))
    estimated_prompt_tokens = max(
        1,
        int(math.ceil((len(controller_text) + len(user_query) + len(original_messages) * 20) / 4.0)),
    )

    metadata = {
        "has_images": has_images,
        "image_count": sum(1 for item in attachments if item.attachment_type == "image"),
        "has_files": has_files,
        "attachment_types": attachment_types,
        "contains_urls": contains_urls,
        "contains_code_blocks": contains_code_blocks,
        "estimated_prompt_tokens": estimated_prompt_tokens,
        "message_count": len(original_messages),
    }

    return RequestState(
        request_id=request_id,
        conversation_id=thread_id,
        thread_id=thread_id,
        model=payload.model,
        stream=payload.stream,
        messages=[ChatMessage.model_validate(message) for message in controller_messages],
        user_message=user_query,
        images=[item.placeholder for item in attachments if item.attachment_type == "image"],
        metadata={
            **metadata,
            "attachments": [item.model_dump(exclude_none=True) for item in attachments],
            "file_count": sum(item.attachment_type != "image" for item in attachments),
        },
    )
