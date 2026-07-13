from __future__ import annotations

import base64
import hashlib
from typing import Any

import httpx

from ..common.enums import ChatRole
from ..settings import Settings
from ..models.vision import ResolvedImage


def collect_latest_message_images(messages: list[dict[str, Any]] | None, max_images: int) -> list[str]:
    if not messages:
        return []

    for message in reversed(messages):
        if message.get("role") != ChatRole.USER.value:
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        refs: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url")
                if isinstance(url, str) and url.strip():
                    refs.append(url.strip())
            elif isinstance(part, dict) and part.get("type") == "image":
                url = part.get("url")
                if isinstance(url, str) and url.strip():
                    refs.append(url.strip())
        return refs[:max_images]
    return []


def extract_latest_user_text(messages: list[dict[str, Any]] | None) -> str:
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
                if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                    parts.append(str(part["text"]).strip())
            return "\n".join(parts).strip()
    return ""


def strip_images_from_messages(messages: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not messages:
        return []

    cleaned: list[dict[str, Any]] = []
    for message in messages:
        message = dict(message)
        content = message.get("content")
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                    text_parts.append(str(part["text"]))
            message["content"] = "\n".join(text_parts).strip() if text_parts else ""
        cleaned.append(message)
    return cleaned


async def resolve_image_ref(
    ref: str,
    *,
    settings: Settings,
    headers: dict[str, str] | None = None,
    client: httpx.AsyncClient | None = None,
) -> ResolvedImage | None:
    raw: bytes | None = None
    mime_type = "image/png"
    source = ref

    try:
        if ref.startswith("data:image/") and ";base64," in ref:
            header, encoded = ref.split(",", 1)
            mime_type = header.split(";", 1)[0].split(":", 1)[1]
            raw = base64.b64decode(encoded)
        elif ref.startswith("http://") or ref.startswith("https://"):
            close_client = False
            if client is None:
                client = httpx.AsyncClient(timeout=settings.vision_timeout_s)
                close_client = True
            try:
                resp = await client.get(ref, headers=headers or {})
                resp.raise_for_status()
                raw = resp.content
                mime_type = resp.headers.get("content-type", mime_type)
            finally:
                if close_client:
                    await client.aclose()
        else:
            return None

        if raw is None:
            return None

        sha256 = hashlib.sha256(raw).hexdigest()
        base64_data = base64.b64encode(raw).decode("utf-8")
        return ResolvedImage(base64_data=base64_data, mime_type=mime_type, sha256=sha256, source=source)
    except Exception:
        return None
