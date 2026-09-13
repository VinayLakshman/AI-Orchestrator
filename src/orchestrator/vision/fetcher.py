from __future__ import annotations

import base64
import hashlib
import logging
from io import BytesIO
from typing import Any

import httpx

from ..logging import get_logger
from ..settings import Settings
from ..models.vision import ResolvedImage


logger = get_logger(__name__)


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


def _image_dimensions(raw: bytes) -> tuple[int, int] | None:
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return None

    try:
        with Image.open(BytesIO(raw)) as image:
            return int(image.width), int(image.height)
    except Exception:
        return None


def _log_image_resolution(source: str, mime_type: str, raw: bytes, base64_data: str) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return

    dimensions = _image_dimensions(raw)
    logger.debug(
        "VISION IMAGE RESOLVED source=%s mime_type=%s raw_size=%d encoded_length=%d dimensions=%s",
        source,
        mime_type,
        len(raw),
        len(base64_data),
        dimensions,
    )


def _prepare_image(raw: bytes, mime_type: str, settings: Settings) -> tuple[bytes, str] | None:
    """Apply soft VRAM/request bounds while preserving image semantics."""
    max_bytes = max(1, int(settings.vision_max_image_bytes))
    max_dimension = max(1, int(settings.vision_max_dimension))
    dimensions = _image_dimensions(raw)
    needs_resize = bool(
        dimensions
        and max(dimensions) > max_dimension
    )
    if len(raw) <= max_bytes and not needs_resize:
        return raw, mime_type.split(";", 1)[0].strip().lower() or "image/png"

    try:
        from PIL import Image  # type: ignore
    except Exception:
        return None

    try:
        with Image.open(BytesIO(raw)) as image:
            image.load()
            source_mime = mime_type.split(";", 1)[0].strip().lower()
            is_photo = source_mime in {"image/jpeg", "image/jpg", "image/webp"}
            has_alpha = "A" in image.getbands()
            target_mime = "image/jpeg" if is_photo and not has_alpha else "image/png"

            width, height = image.size
            scale = min(1.0, max_dimension / max(width, height))
            if len(raw) > max_bytes:
                scale = min(scale, 0.85)
            current = image.copy()
            for _ in range(5):
                target_size = (
                    max(1, int(width * scale)),
                    max(1, int(height * scale)),
                )
                if target_size != current.size:
                    current = current.resize(target_size, Image.Resampling.LANCZOS)
                output = BytesIO()
                if target_mime == "image/jpeg":
                    current.convert("RGB").save(output, format="JPEG", quality=85, optimize=True)
                else:
                    current.save(output, format="PNG", optimize=True)
                candidate = output.getvalue()
                if len(candidate) <= max_bytes and max(candidate and target_size) <= max_dimension:
                    return candidate, target_mime
                scale *= 0.75
            return None
    except Exception:
        return None


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
            if len(encoded) * 3 // 4 > int(settings.vision_max_image_bytes) * 2:
                return None
            mime_type = header.split(";", 1)[0].split(":", 1)[1]
            raw = base64.b64decode(encoded, validate=False)
        elif ref.startswith("http://") or ref.startswith("https://"):
            close_client = False
            if client is None:
                client = httpx.AsyncClient(timeout=settings.vision_timeout_s)
                close_client = True
            try:
                async with client.stream("GET", ref, headers=headers or {}) as resp:
                    resp.raise_for_status()
                    content_length = resp.headers.get("content-length")
                    max_download = int(settings.vision_max_image_bytes) * 2
                    if content_length and int(content_length) > max_download:
                        return None
                    chunks: list[bytes] = []
                    downloaded = 0
                    async for chunk in resp.aiter_bytes():
                        downloaded += len(chunk)
                        if downloaded > max_download:
                            return None
                        chunks.append(chunk)
                    raw = b"".join(chunks)
                    mime_type = resp.headers.get("content-type", mime_type)
            finally:
                if close_client:
                    await client.aclose()
        else:
            return None

        if raw is None:
            return None

        prepared = _prepare_image(raw, mime_type, settings)
        if prepared is None:
            return None
        raw, mime_type = prepared

        sha256 = hashlib.sha256(raw).hexdigest()
        base64_data = base64.b64encode(raw).decode("utf-8")
        _log_image_resolution(source, mime_type, raw, base64_data)
        return ResolvedImage(base64_data=base64_data, mime_type=mime_type, sha256=sha256, source=source)
    except Exception:
        return None
