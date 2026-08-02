from __future__ import annotations

import copy
import math
import re
from typing import Any

from .common.enums import ChatRole
from .logging import get_logger
from .models.chat import ChatMessage
from .models.state import RequestState
from .schemas import (
    NormalizedAttachment,
    OpenAIChatCompletionRequest,
    OpenAIMessage,
)

logger = get_logger(__name__)

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_CODE_BLOCK_RE = re.compile(r"```", re.DOTALL)

_TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".rst",
    ".py",
    ".java",
    ".kt",
    ".groovy",
    ".scala",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".css",
    ".scss",
    ".xml",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".sql",
    ".properties",
    ".ini",
    ".cfg",
    ".conf",
    ".env",
    ".gradle",
    ".dockerfile",
    ".sh",
    ".bash",
    ".zsh",
    ".ps1",
    ".log",
    ".out",
    ".trace",
    ".csv",
}

_RICH_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".pptx"}
_HTML_EXTENSIONS = {".html", ".htm"}
_ARCHIVE_SUFFIXES = (".zip", ".tar", ".tgz", ".tar.gz")


def _is_image_part(part: dict[str, Any]) -> bool:
    part_type = str(part.get("type") or "").lower()
    if part_type in {"image_url", "image", "input_image"}:
        return True
    mime_type = ""
    image_url = part.get("image_url")
    if isinstance(image_url, dict):
        mime_type = str(image_url.get("url") or "")
    elif isinstance(image_url, str):
        mime_type = image_url
    if isinstance(part.get("mime_type"), str):
        mime_type = part["mime_type"]
    return "image" in part_type or mime_type.startswith("data:image/")


def _is_file_part(part: dict[str, Any]) -> bool:
    part_type = str(part.get("type") or "").lower()
    if part_type in {"file", "input_file", "document", "attachment", "pdf", "archive"}:
        return True
    filename = str(part.get("filename") or part.get("name") or part.get("path") or "").lower()
    mime_type = str(part.get("mime_type") or part.get("media_type") or "").lower()
    if filename.endswith(_ARCHIVE_SUFFIXES):
        return True
    if filename.endswith(tuple(_TEXT_EXTENSIONS)) or filename.endswith(tuple(_RICH_EXTENSIONS)) or filename.endswith(tuple(_HTML_EXTENSIONS)):
        return True
    return mime_type.startswith(("application/", "text/"))


def _attachment_type(part: dict[str, Any]) -> str:
    if _is_image_part(part):
        return "image"
    if _is_file_part(part):
        filename = str(part.get("filename") or part.get("name") or part.get("path") or "").lower()
        mime_type = str(part.get("mime_type") or part.get("media_type") or "").lower()
        if filename.endswith(_ARCHIVE_SUFFIXES):
            return "archive"
        if filename.endswith(".pdf") or "pdf" in mime_type:
            return "pdf"
        if filename.endswith(tuple(_HTML_EXTENSIONS)) or "html" in mime_type:
            return "html"
        if filename.endswith(tuple(_RICH_EXTENSIONS)) or any(filename.endswith(ext) for ext in _TEXT_EXTENSIONS):
            return "document"
        if filename.endswith(tuple(_TEXT_EXTENSIONS)) or mime_type.startswith("text/"):
            return "text"
        return "file"
    return str(part.get("type") or "attachment").lower()


def _placeholder_for_attachment(attachment_type: str) -> str:
    return {
        "image": "<Image Attached>",
        "pdf": "<PDF Attached>",
        "document": "<Document Attached>",
        "file": "<File Attached>",
    }.get(attachment_type, "<Attachment Attached>")


def _extract_attachment_reference(part: dict[str, Any]) -> str:
    image_url = part.get("image_url")
    if isinstance(image_url, str) and image_url.strip():
        return image_url.strip()
    if isinstance(image_url, dict):
        for key in ("url", "source", "path", "data"):
            value = image_url.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for key in ("url", "source", "path", "file_id", "filename", "name"):
        value = part.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _attachment_reference(part: dict[str, Any], attachment_type: str) -> str:
    reference = _extract_attachment_reference(part)
    if reference:
        return reference
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

    # Always set user_query from the *latest* user message.
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
        if message.role == ChatRole.USER.value and content and content.strip():
            user_query = content

    # Fallback: if no non-empty content was found, pick the latest user message.
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

    logger.debug(
        "NORMALIZE: request_id=%s thread_id=%s user_message_len=%d has_images=%s image_count=%d image_ref_kinds=%s message_count=%d user_message_preview=%r",
        request_id,
        thread_id,
        len(user_query or ""),
        any(item.attachment_type == "image" for item in attachments),
        sum(1 for item in attachments if item.attachment_type == "image"),
        [
            "data_uri"
            if item.placeholder.startswith("data:image/")
            else "http_url"
            if item.placeholder.startswith(("http://", "https://"))
            else "placeholder"
            if item.placeholder.startswith("<")
            else "other"
            for item in attachments
            if item.attachment_type == "image"
        ],
        len(original_messages),
        (user_query or "")[:120],
    )

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
        original_query=user_query,
        resolved_query=user_query,
        is_followup=False,
        followup_confidence=0.0,
        images=[item.placeholder for item in attachments if item.attachment_type == "image"],
        metadata={
            **metadata,
            "attachments": [item.model_dump(exclude_none=True) for item in attachments],
            "file_count": sum(item.attachment_type != "image" for item in attachments),
        },
    )
