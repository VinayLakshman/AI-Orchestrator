from __future__ import annotations

import base64
import copy
import hashlib
from typing import Any
from urllib.parse import urljoin

import httpx

from ..settings import Settings
from .models import ResolvedImage


def _extract_ref_string(ref: Any) -> str:
    if isinstance(ref, str):
        return ref.strip()

    if isinstance(ref, dict):
        for key in ("url", "image", "content", "path", "base64", "data", "src"):
            value = ref.get(key)
            if value:
                return str(value).strip()

    return ""


def _looks_like_image_filename(name: str) -> bool:
    name = (name or "").lower().strip()
    return name.endswith(
        (
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".gif",
            ".bmp",
            ".svg",
            ".tif",
            ".tiff",
            ".heic",
            ".heif",
        )
    )


def _is_probably_image_file(item: Any) -> bool:
    if isinstance(item, str):
        s = item.strip().lower()
        if not s:
            return False
        if s.startswith("data:image/"):
            return True
        if _looks_like_image_filename(s):
            return True
        return False

    if not isinstance(item, dict):
        return False

    for key in ("content_type", "mime_type"):
        v = str(item.get(key, "")).lower().strip()
        if v.startswith("image/"):
            return True

    t = str(item.get("type", "")).lower().strip()
    if t in ("image", "image_url", "input_image"):
        return True

    for key in ("name", "filename", "file_name"):
        v = str(item.get(key, "")).lower().strip()
        if _looks_like_image_filename(v):
            return True

    for key in ("url", "path", "content", "image", "src"):
        v = str(item.get(key, "")).strip()
        if v.startswith("data:image/"):
            return True
        if _looks_like_image_filename(v):
            return True

    return False


def normalize_data_url(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return raw
    if raw.startswith("data:"):
        return raw
    return f"data:image/png;base64,{raw}"


def get_latest_user_message(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for msg in reversed(messages or []):
        if msg.get("role") == "user":
            return msg
    return None


def collect_latest_message_images(messages: list[dict[str, Any]], max_images: int) -> list[str]:
    """
    Collect only images from the latest user message.

    Supports:
    - OpenAI-style content parts: [{"type":"image_url", ...}, ...]
    - OpenWebUI-style content parts: [{"type":"image", ...}, ...]
    - images list
    - files list (only if they look like images)
    """
    latest = get_latest_user_message(messages)
    if latest is None:
        return []

    images: list[str] = []
    seen: set[str] = set()

    def add_ref(raw: Any) -> None:
        ref = _extract_ref_string(raw)
        if not ref:
            return
        if ref not in seen:
            seen.add(ref)
            images.append(ref)

    content = latest.get("content")

    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue

            t = str(part.get("type", "")).lower().strip()

            if t == "image_url":
                img = part.get("image_url")
                if isinstance(img, dict):
                    add_ref(img.get("url"))
                else:
                    add_ref(img)

            elif t in ("image", "input_image"):
                add_ref(part.get("url"))
                add_ref(part.get("image"))
                add_ref(part.get("content"))
                add_ref(part.get("src"))

    for img in latest.get("images", []) or []:
        add_ref(img)

    for file_item in latest.get("files", []) or []:
        if _is_probably_image_file(file_item):
            add_ref(file_item)

    return images[:max_images]


def extract_latest_user_text(messages: list[dict[str, Any]]) -> str:
    latest = get_latest_user_message(messages)
    if latest is None:
        return ""

    content = latest.get("content", "")
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = str(part.get("text", "")).strip()
                if text:
                    text_parts.append(text)
        return "\n".join(text_parts).strip()

    return ""


def strip_images_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Remove all image payloads from messages so the downstream text model only receives text.
    """
    cleaned = copy.deepcopy(messages)

    for msg in cleaned:
        msg.pop("images", None)
        msg.pop("files", None)

        content = msg.get("content", "")

        if isinstance(content, list):
            text_only = []
            for part in content:
                if not isinstance(part, dict):
                    continue

                t = str(part.get("type", "")).lower().strip()

                if t in ("image", "image_url", "input_image", "file"):
                    continue

                if t == "text":
                    text = str(part.get("text", "")).strip()
                    if text:
                        text_only.append({"type": "text", "text": text})
                    continue

                if t:
                    text_value = part.get("text")
                    if isinstance(text_value, str) and text_value.strip():
                        text_only.append({"type": "text", "text": text_value.strip()})

            if len(text_only) == 1:
                msg["content"] = text_only[0]["text"]
            else:
                msg["content"] = "\n".join(part["text"] for part in text_only if part.get("text")).strip()

        for k in ("attachments", "media", "multimodal", "image"):
            if k in msg and isinstance(msg[k], (list, dict, str)):
                msg.pop(k, None)

    return cleaned


def _decode_data_url(ref: str) -> tuple[str, bytes]:
    """
    Returns (mime_type, raw_bytes).
    """
    header, _, payload = ref.partition(",")
    mime_type = "image/png"
    if header.startswith("data:"):
        before_base64 = header[5:]
        mime_part = before_base64.split(";", 1)[0].strip()
        if mime_part:
            mime_type = mime_part
    raw = base64.b64decode(payload or "", validate=False)
    return mime_type, raw


async def resolve_image_ref(
    ref: Any,
    *,
    settings: Settings,
    headers: dict[str, str] | None,
    client: httpx.AsyncClient,
) -> ResolvedImage | None:
    """
    Convert supported image references to raw base64 for Ollama.
    Supported:
    - data URLs
    - raw base64 strings
    - relative / absolute URLs that can be fetched
    """
    raw_ref = _extract_ref_string(ref)
    if not raw_ref:
        return None

    mime_type = "image/png"
    raw_bytes: bytes = b""

    try:
        if raw_ref.startswith("data:"):
            mime_type, raw_bytes = _decode_data_url(raw_ref)

        elif raw_ref.startswith("http://") or raw_ref.startswith("https://") or raw_ref.startswith("/"):
            if raw_ref.startswith("/"):
                base = settings.vision_fetch_base_url.rstrip("/") + "/"
                full_url = urljoin(base, raw_ref.lstrip("/"))
            else:
                full_url = raw_ref

            resp = await client.get(full_url, headers=headers or {})
            resp.raise_for_status()

            mime_type = resp.headers.get("content-type", "image/png").split(";", 1)[0].strip() or "image/png"
            raw_bytes = resp.content

        else:
            # Assume raw base64
            raw_bytes = base64.b64decode(raw_ref, validate=False)
            mime_type = "image/png"

    except Exception:
        return None

    if not raw_bytes:
        return None

    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    base64_data = base64.b64encode(raw_bytes).decode("utf-8")

    return ResolvedImage(
        ref=raw_ref,
        mime_type=mime_type,
        sha256=sha256,
        base64_data=base64_data,
    )