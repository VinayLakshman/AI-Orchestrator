from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Iterable

from ..artifact import DocumentArtifact, DocumentCategory, RepositoryArtifact


@dataclass(slots=True)
class ArchiveMember:
    filename: str
    raw: bytes
    size: int


def _tree_from_paths(paths: Iterable[str]) -> dict[str, Any]:
    root: dict[str, Any] = {}
    for path in paths:
        current = root
        parts = [part for part in path.split("/") if part]
        for part in parts:
            current = current.setdefault(part, {})
    return root


def _extension_for_member(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".tar.gz"):
        return ".tar.gz"
    if lower.endswith(".tgz"):
        return ".tgz"
    return os.path.splitext(lower)[1]


def iter_zip_members(raw: bytes) -> list[ArchiveMember]:
    import zipfile

    members: list[ArchiveMember] = []
    with zipfile.ZipFile(BytesIO(raw)) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            with archive.open(info) as handle:
                members.append(ArchiveMember(filename=info.filename, raw=handle.read(), size=info.file_size))
    return members


def iter_tar_members(raw: bytes) -> list[ArchiveMember]:
    import tarfile

    members: list[ArchiveMember] = []
    with tarfile.open(fileobj=BytesIO(raw), mode="r:*") as archive:
        for info in archive.getmembers():
            if not info.isfile():
                continue
            handle = archive.extractfile(info)
            if handle is None:
                continue
            members.append(ArchiveMember(filename=info.name, raw=handle.read(), size=info.size))
    return members


def build_repository_artifact(
    *,
    root: str,
    documents: list[DocumentArtifact],
    metadata: dict[str, Any] | None = None,
) -> RepositoryArtifact:
    paths = [doc.filename for doc in documents if doc.filename]
    tree = _tree_from_paths(paths)
    safe_metadata = {}
    if metadata:
        for key in ("source", "mime_type", "checksum", "member_count", "nested_repository_count"):
            if key in metadata:
                safe_metadata[key] = metadata[key]
    return RepositoryArtifact(
        root=root,
        documents=documents,
        directory_tree=tree,
        metadata=safe_metadata,
        statistics={
            "document_count": len(documents),
            "text_documents": sum(1 for doc in documents if doc.category == DocumentCategory.TEXT),
            "rich_documents": sum(1 for doc in documents if doc.category == DocumentCategory.RICH),
            "html_documents": sum(1 for doc in documents if doc.category == DocumentCategory.HTML),
        },
    )

