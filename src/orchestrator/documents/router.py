from __future__ import annotations

import mimetypes
import os
from dataclasses import dataclass

from .artifact import DocumentCategory


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
    ".html",
    ".htm",
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
    ".sh",
    ".bash",
    ".zsh",
    ".ps1",
    ".log",
    ".out",
    ".trace",
    ".csv",
}

_ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".zip", ".tar")

_RICH_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".xlsx",
    ".pptx",
}

_HTML_EXTENSIONS = {".html", ".htm"}


@dataclass(slots=True)
class AttachmentRoute:
    category: DocumentCategory
    extension: str
    mime_type: str
    is_archive: bool = False
    is_image: bool = False


def normalize_extension(filename: str) -> str:
    name = filename.lower().strip()
    for suffix in _ARCHIVE_SUFFIXES:
        if name.endswith(suffix):
            return suffix
    return os.path.splitext(name)[1]


def guess_mime_type(filename: str, mime_type: str | None = None) -> str:
    if mime_type and str(mime_type).strip():
        return str(mime_type).strip().lower()
    guessed, _ = mimetypes.guess_type(filename)
    return (guessed or "application/octet-stream").lower()


def route_attachment(*, filename: str, mime_type: str | None = None) -> AttachmentRoute:
    extension = normalize_extension(filename)
    guessed_mime = guess_mime_type(filename, mime_type)

    if guessed_mime.startswith("image/"):
        return AttachmentRoute(
            category=DocumentCategory.IMAGE,
            extension=extension,
            mime_type=guessed_mime,
            is_image=True,
        )

    if extension in _ARCHIVE_SUFFIXES:
        return AttachmentRoute(
            category=DocumentCategory.ARCHIVE,
            extension=extension,
            mime_type=guessed_mime,
            is_archive=True,
        )

    if extension in _RICH_EXTENSIONS:
        return AttachmentRoute(category=DocumentCategory.RICH, extension=extension, mime_type=guessed_mime)

    if extension in _HTML_EXTENSIONS or "html" in guessed_mime:
        return AttachmentRoute(category=DocumentCategory.HTML, extension=extension, mime_type=guessed_mime)

    if extension in _TEXT_EXTENSIONS or guessed_mime.startswith(("text/", "application/json", "application/xml")):
        return AttachmentRoute(category=DocumentCategory.TEXT, extension=extension, mime_type=guessed_mime)

    return AttachmentRoute(category=DocumentCategory.UNSUPPORTED, extension=extension, mime_type=guessed_mime)

