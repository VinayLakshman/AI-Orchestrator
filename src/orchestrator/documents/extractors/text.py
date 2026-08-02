from __future__ import annotations

import hashlib
import re
from typing import Any

from ..artifact import DocumentArtifact, DocumentCategory


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _line_count(text: str) -> int:
    if not text:
        return 0
    return text.count("\n") + 1


def extract_text_document(
    *,
    raw: bytes,
    filename: str,
    mime_type: str,
    encoding: str = "utf-8",
    language: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> DocumentArtifact:
    checksum = hashlib.sha256(raw).hexdigest()
    text = ""
    used_encoding = encoding or "utf-8"
    for candidate in (used_encoding, "utf-8-sig", "utf-8", "utf-16", "latin-1"):
        try:
            text = raw.decode(candidate)
            used_encoding = candidate
            break
        except Exception:
            continue

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return DocumentArtifact(
        id=checksum[:16],
        filename=filename,
        extension=filename.lower().rpartition(".")[-1] if "." in filename else "",
        mime_type=mime_type,
        size=len(raw),
        checksum=checksum,
        category=DocumentCategory.TEXT,
        language=language,
        encoding=used_encoding,
        content_type="text",
        metadata=dict(metadata or {}),
        text=text,
        line_count=_line_count(text),
        page_count=0,
        word_count=_word_count(text),
        status="ok",
    )

