from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class DocumentCategory(StrEnum):
    TEXT = "text"
    RICH = "rich"
    HTML = "html"
    ARCHIVE = "archive"
    IMAGE = "image"
    UNSUPPORTED = "unsupported"


class DocumentPage(BaseModel):
    number: int = 0
    text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentTable(BaseModel):
    name: str = ""
    text: str = ""
    rows: list[list[str]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentSection(BaseModel):
    title: str = ""
    text: str = ""
    start_line: int | None = None
    end_line: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentImageReference(BaseModel):
    reference: str = ""
    description: str = ""
    page: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentArtifact(BaseModel):
    id: str = ""
    filename: str = ""
    extension: str = ""
    mime_type: str = ""
    size: int = 0
    checksum: str = ""
    category: DocumentCategory = DocumentCategory.UNSUPPORTED
    language: str | None = None
    encoding: str = "utf-8"
    content_type: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    text: str = ""
    pages: list[DocumentPage] = Field(default_factory=list)
    tables: list[DocumentTable] = Field(default_factory=list)
    sections: list[DocumentSection] = Field(default_factory=list)
    images: list[DocumentImageReference] = Field(default_factory=list)
    line_count: int = 0
    page_count: int = 0
    word_count: int = 0
    status: str = "ok"
    error: str = ""

    @property
    def is_supported(self) -> bool:
        return self.status == "ok" and self.category != DocumentCategory.UNSUPPORTED


class RepositoryArtifact(BaseModel):
    root: str = ""
    documents: list[DocumentArtifact] = Field(default_factory=list)
    directory_tree: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    statistics: dict[str, Any] = Field(default_factory=dict)

