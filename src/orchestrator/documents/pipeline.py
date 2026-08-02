from __future__ import annotations

import base64
import hashlib
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..logging import get_logger
from ..models.evidence import DocumentEvidence
from ..settings import Settings
from .artifact import DocumentArtifact, DocumentCategory, DocumentPage, DocumentSection, RepositoryArtifact

MAX_ARCHIVE_DEPTH = 3
MAX_ARCHIVE_MEMBERS = 256
from .extractors.archive import ArchiveMember, build_repository_artifact, iter_tar_members, iter_zip_members
from .extractors.html import extract_html_document
from .extractors.office import extract_docx_document, extract_pptx_document, extract_xlsx_document
from .extractors.pdf import extract_pdf_document
from .extractors.text import extract_text_document
from .router import AttachmentRoute, route_attachment

logger = get_logger(__name__)


def _checksum(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _line_count(text: str) -> int:
    if not text:
        return 0
    return text.count("\n") + 1


def _safe_filename(value: str | None, default: str = "attachment") -> str:
    text = str(value or "").strip()
    if not text:
        return default
    text = text.split("?", 1)[0].split("#", 1)[0]
    return os.path.basename(text) or default


def _attachment_value(raw: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _looks_like_data_uri(value: str) -> bool:
    return value.startswith("data:")


def _decode_data_uri(value: str) -> tuple[bytes, str]:
    header, encoded = value.split(",", 1)
    mime_type = header.split(";", 1)[0].split(":", 1)[1] if ":" in header else "application/octet-stream"
    return base64.b64decode(encoded), mime_type


def _infer_language(filename: str, extension: str) -> str | None:
    lower = filename.lower()
    mapping = {
        ".py": "python",
        ".java": "java",
        ".kt": "kotlin",
        ".groovy": "groovy",
        ".scala": "scala",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".sh": "shell",
        ".bash": "shell",
        ".zsh": "shell",
        ".ps1": "powershell",
        ".sql": "sql",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".html": "html",
        ".htm": "html",
        ".css": "css",
        ".scss": "css",
        ".xml": "xml",
        ".csv": "csv",
    }
    if extension in mapping:
        return mapping[extension]
    if lower.endswith("dockerfile") or os.path.basename(lower) == "dockerfile":
        return "dockerfile"
    if lower.endswith("compose.yaml") or lower.endswith("docker-compose.yml"):
        return "yaml"
    return None


def _sections_from_text(text: str) -> list[DocumentSection]:
    sections: list[DocumentSection] = []
    for index, block in enumerate(re.split(r"\n{2,}", text.strip()), start=1):
        cleaned = block.strip()
        if not cleaned:
            continue
        sections.append(
            DocumentSection(
                title=f"section_{index}",
                text=cleaned,
                metadata={"section_index": index},
            )
        )
    return sections


def _text_excerpts(document: DocumentArtifact) -> list[str]:
    excerpts: list[str] = []
    if document.pages:
        for page in document.pages[:4]:
            text = page.text.strip()
            if text:
                excerpts.append(text[:400])
    if not excerpts and document.sections:
        for section in document.sections[:4]:
            text = section.text.strip()
            if text:
                excerpts.append(text[:400])
    if not excerpts and document.text.strip():
        for line in document.text.splitlines():
            line = line.strip()
            if line:
                excerpts.append(line[:400])
            if len(excerpts) >= 4:
                break
    return [excerpt for excerpt in excerpts if excerpt.strip()]


def _artifact_summary(document: DocumentArtifact) -> dict[str, Any]:
    return {
        "id": document.id,
        "filename": document.filename,
        "extension": document.extension,
        "mime_type": document.mime_type,
        "category": document.category.value if hasattr(document.category, "value") else str(document.category),
        "language": document.language,
        "encoding": document.encoding,
        "size": document.size,
        "checksum": document.checksum,
        "line_count": document.line_count,
        "page_count": document.page_count,
        "word_count": document.word_count,
        "status": document.status,
        "error": document.error,
        "metadata": document.metadata,
        "excerpts": _text_excerpts(document),
    }


def _failure_artifact(
    *,
    filename: str,
    mime_type: str,
    content_type: str,
    size: int,
    error: str,
    checksum: str = "",
    exception: Exception | None = None,
) -> DocumentArtifact:
    resolved_checksum = checksum or hashlib.sha256(f"{filename}:{error}".encode()).hexdigest()
    metadata = _canonical_document_metadata(
        filename=filename,
        extension=route_attachment(filename=filename, mime_type=mime_type).extension,
        mime_type=mime_type,
        size=size,
        checksum=resolved_checksum,
        route=route_attachment(filename=filename, mime_type=mime_type),
    )
    metadata.update({"status": "failed", "error": error})
    return DocumentArtifact(
        id=resolved_checksum[:16],
        filename=filename,
        extension=route_attachment(filename=filename, mime_type=mime_type).extension,
        mime_type=mime_type,
        size=size,
        checksum=resolved_checksum,
        category=DocumentCategory.UNSUPPORTED,
        encoding="utf-8",
        content_type=content_type,
        metadata=metadata,
        text="",
        pages=[],
        tables=[],
        sections=[],
        images=[],
        line_count=0,
        page_count=0,
        word_count=0,
        status="failed",
        error=str(exception) if exception else error,
    )


def _text_artifact(
    *,
    raw: bytes,
    filename: str,
    mime_type: str,
    metadata: dict[str, Any] | None = None,
    language: str | None = None,
    category: DocumentCategory = DocumentCategory.TEXT,
) -> DocumentArtifact:
    artifact = extract_text_document(
        raw=raw,
        filename=filename,
        mime_type=mime_type,
        language=language or _infer_language(filename, route_attachment(filename=filename, mime_type=mime_type).extension),
        metadata=metadata,
    )
    artifact.category = category
    artifact.content_type = category.value
    if not artifact.sections and artifact.text.strip():
        artifact.sections = _sections_from_text(artifact.text)
    if not artifact.pages and artifact.text.strip():
        artifact.pages = [DocumentPage(number=1, text=artifact.text, metadata={"source": "text"})]
        artifact.page_count = 1
    artifact.line_count = artifact.line_count or _line_count(artifact.text)
    artifact.word_count = artifact.word_count or _word_count(artifact.text)
    artifact.metadata = _canonical_document_metadata(
        filename=artifact.filename,
        extension=artifact.extension,
        mime_type=artifact.mime_type,
        size=artifact.size,
        checksum=artifact.checksum,
        route=route_attachment(filename=artifact.filename, mime_type=artifact.mime_type),
    )
    return artifact


def _extract_leaf_document(
    *,
    raw: bytes,
    filename: str,
    mime_type: str,
    route: AttachmentRoute,
    metadata: dict[str, Any] | None = None,
) -> DocumentArtifact:
    canonical_metadata = _canonical_document_metadata(
        filename=filename,
        extension=route.extension,
        mime_type=mime_type,
        size=len(raw),
        checksum=_checksum(raw),
        route=route,
    )
    if route.category == DocumentCategory.TEXT:
        artifact = _text_artifact(raw=raw, filename=filename, mime_type=mime_type, metadata=canonical_metadata)
    elif route.category == DocumentCategory.HTML:
        artifact = extract_html_document(raw=raw, filename=filename, mime_type=mime_type, metadata=canonical_metadata)
    elif route.extension == ".pdf":
        artifact = extract_pdf_document(raw=raw, filename=filename, mime_type=mime_type, metadata=canonical_metadata)
    elif route.extension == ".docx":
        artifact = extract_docx_document(raw=raw, filename=filename, mime_type=mime_type, metadata=canonical_metadata)
    elif route.extension == ".xlsx":
        artifact = extract_xlsx_document(raw=raw, filename=filename, mime_type=mime_type, metadata=canonical_metadata)
    elif route.extension == ".pptx":
        artifact = extract_pptx_document(raw=raw, filename=filename, mime_type=mime_type, metadata=canonical_metadata)
    else:
        return _failure_artifact(
            filename=filename,
            mime_type=mime_type,
            content_type=route.category.value,
            size=len(raw),
            error="unsupported_document_format",
            checksum=_checksum(raw),
        )

    # Ensure metadata stays canonical after extraction.
    artifact.metadata = _canonical_document_metadata(
        filename=artifact.filename,
        extension=artifact.extension,
        mime_type=artifact.mime_type,
        size=artifact.size,
        checksum=artifact.checksum,
        route=route,
        encoding=artifact.encoding,
        language=artifact.language,
        page_count=artifact.page_count,
        line_count=artifact.line_count,
        word_count=artifact.word_count,
    )
    return artifact


async def _resolve_attachment_bytes(
    attachment: dict[str, Any],
    *,
    settings: Settings,
    headers: dict[str, str] | None = None,
    client: httpx.AsyncClient | None = None,
) -> tuple[bytes | None, str, str]:
    filename = _safe_filename(_attachment_value(attachment, "filename", "name", "title", "path"), default="attachment")
    mime_type = _attachment_value(attachment, "mime_type", "media_type", "content_type") or ""

    source = _attachment_value(
        attachment,
        "url",
        "source",
        "path",
        "file_url",
        "download_url",
        "content_url",
        "data",
        "file",
    )

    if not source and isinstance(attachment.get("image_url"), dict):
        source = _attachment_value(attachment["image_url"], "url", "source", "path", "data")
        mime_type = mime_type or _attachment_value(attachment["image_url"], "mime_type", "media_type")
    elif not source and isinstance(attachment.get("image_url"), str):
        source = attachment["image_url"].strip()

    if not source:
        return None, filename, mime_type or "application/octet-stream"

    if _looks_like_data_uri(source):
        raw, detected_mime = _decode_data_uri(source)
        return raw, filename, mime_type or detected_mime

    if source.startswith(("http://", "https://")):
        close_client = False
        if client is None:
            client = httpx.AsyncClient(timeout=settings.request_timeout_s)
            close_client = True
        try:
            response = await client.get(source, headers=headers or {})
            response.raise_for_status()
            detected_mime = response.headers.get("content-type", mime_type or "application/octet-stream")
            return response.content, filename, detected_mime
        finally:
            if close_client:
                await client.aclose()

    if os.path.exists(source):
        with open(source, "rb") as handle:
            return handle.read(), filename, mime_type or "application/octet-stream"

    if len(source) > 128 and "=" in source:
        try:
            return base64.b64decode(source), filename, mime_type or "application/octet-stream"
        except Exception:
            pass

    return None, filename, mime_type or "application/octet-stream"


def _split_archive_members(raw: bytes, filename: str) -> list[ArchiveMember]:
    lower = filename.lower()
    if lower.endswith(".zip"):
        return iter_zip_members(raw, max_members=MAX_ARCHIVE_MEMBERS)
    if lower.endswith((".tar", ".tgz", ".tar.gz")):
        return iter_tar_members(raw, max_members=MAX_ARCHIVE_MEMBERS)
    return []


def _canonical_document_metadata(
    *,
    filename: str,
    extension: str,
    mime_type: str,
    size: int,
    checksum: str,
    route: AttachmentRoute | None = None,
    encoding: str | None = None,
    language: str | None = None,
    page_count: int | None = None,
    line_count: int | None = None,
    word_count: int | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "filename": filename,
        "extension": extension,
        "mime_type": mime_type,
        "size": size,
        "checksum": checksum,
        "encoding": encoding,
        "language": language,
        "page_count": page_count,
        "line_count": line_count,
        "word_count": word_count,
    }
    if route is not None:
        metadata["content_type"] = route.category.value
        metadata["category"] = route.category.value
    return {key: value for key, value in metadata.items() if value is not None and value != ""}


def _chunk_text(text: str, *, chunk_size: int = 2200) -> list[str]:
    lines = [line.rstrip() for line in text.splitlines()]
    chunks: list[str] = []
    window: list[str] = []
    current = 0
    for line in lines:
        if not line.strip():
            if window:
                window.append("")
            continue
        if current + len(line) > chunk_size and window:
            chunk = "\n".join(window).strip()
            if chunk:
                chunks.append(chunk)
            window = []
            current = 0
        window.append(line)
        current += len(line)
    if window:
        chunk = "\n".join(window).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _truncate(text: str, limit: int = 260) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 1)].rstrip() + "…"


def _normalize_query_terms(query: str) -> list[str]:
    return [term.lower() for term in re.findall(r"\w+", query) if len(term) > 1]


def _score_chunk(text: str, terms: list[str], boost: float = 0.0) -> float:
    if not terms:
        return boost + (0.1 if text.strip() else 0.0)
    haystack = text.lower()
    score = sum(haystack.count(term) for term in terms)
    return float(score) + boost


def _collect_document_chunks(documents: list[DocumentArtifact]) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for document in documents:
        if document.pages:
            for page in document.pages:
                text = str(page.text or "").strip()
                if not text:
                    continue
                chunks.append(
                    DocumentChunk(
                        document_id=document.id,
                        filename=document.filename,
                        text=text,
                        score_hint=1.5,
                        metadata={"page": page.number, **page.metadata},
                    )
                )
            continue

        if document.sections:
            for section in document.sections:
                text = str(section.text or "").strip()
                if not text:
                    continue
                chunks.append(
                    DocumentChunk(
                        document_id=document.id,
                        filename=document.filename,
                        text=text,
                        score_hint=1.0,
                        metadata={"section": section.title, **section.metadata},
                    )
                )
            continue

        if document.text.strip():
            for chunk_text in _chunk_text(document.text):
                chunks.append(
                    DocumentChunk(
                        document_id=document.id,
                        filename=document.filename,
                        text=chunk_text,
                        score_hint=0.5,
                        metadata={"line_count": _line_count(chunk_text), "source": "text"},
                    )
                )
    return chunks


def _build_document_context(
    documents: list[DocumentArtifact],
    query: str | None,
    *,
    max_chars: int = 1800,
    max_chunks: int = 8,
) -> tuple[str, list[str]]:
    terms = _normalize_query_terms(query or "")
    chunks = _collect_document_chunks(documents)
    if not chunks:
        return "", []

    scored = [(_score_chunk(chunk.text, terms, chunk.score_hint), chunk) for chunk in chunks]
    scored.sort(key=lambda item: item[0], reverse=True)
    selected: list[tuple[DocumentChunk, str]] = []
    seen: set[str] = set()
    total_chars = 0

    for score, chunk in scored:
        if not chunk.text.strip():
            continue
        excerpt = _truncate(chunk.text, 260)
        if not excerpt or excerpt in seen:
            continue
        if total_chars + len(excerpt) > max_chars and selected:
            break
        selected.append((chunk, excerpt))
        seen.add(excerpt)
        total_chars += len(excerpt)
        if len(selected) >= max_chunks:
            break

    if not selected:
        return "", []

    lines: list[str] = []
    for chunk, excerpt in selected:
        descriptor = []
        if chunk.filename:
            descriptor.append(f"File: {chunk.filename}")
        if chunk.metadata.get("page") is not None:
            descriptor.append(f"Page: {chunk.metadata.get('page')}")
        if chunk.metadata.get("section") is not None:
            descriptor.append(f"Section: {chunk.metadata.get('section')}")
        if descriptor:
            lines.append(" | ".join(descriptor))
        lines.append(excerpt)
        lines.append("")

    return "\n".join(lines).strip(), [excerpt for _, excerpt in selected]


@dataclass(slots=True)
class DocumentChunk:
    document_id: str
    filename: str
    text: str
    score_hint: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DocumentSession:
    session_id: str
    documents: list[DocumentArtifact] = field(default_factory=list)
    repository_artifacts: list[RepositoryArtifact] = field(default_factory=list)
    chunks: list[DocumentChunk] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    statistics: dict[str, Any] = field(default_factory=dict)
    last_accessed: float = field(default_factory=time.time)

    def search(self, query: str, *, limit: int = 8) -> list[dict[str, Any]]:
        terms = [term.lower() for term in re.findall(r"\w+", query) if len(term) > 1]
        if not terms:
            return []

        scored: list[tuple[float, DocumentChunk]] = []
        for chunk in self.chunks:
            haystack = chunk.text.lower()
            score = sum(haystack.count(term) for term in terms)
            if not score:
                continue
            scored.append((float(score + chunk.score_hint), chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        results: list[dict[str, Any]] = []
        for score, chunk in scored[:limit]:
            results.append(
                {
                    "document_id": chunk.document_id,
                    "filename": chunk.filename,
                    "score": score,
                    "excerpt": chunk.text[:1000],
                    "metadata": chunk.metadata,
                }
            )
        return results


@dataclass(slots=True)
class DocumentPipeline:
    settings: Settings
    client: httpx.AsyncClient | None = None
    _cache: dict[str, DocumentSession] = field(default_factory=dict)

    def _session(self, session_id: str) -> DocumentSession:
        return self._cache.setdefault(session_id, DocumentSession(session_id=session_id))

    async def _extract_archive(
        self,
        *,
        raw: bytes,
        filename: str,
        mime_type: str,
        headers: dict[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
        depth: int = 0,
    ) -> tuple[RepositoryArtifact, list[DocumentArtifact], list[RepositoryArtifact]]:
        members = _split_archive_members(raw, filename)
        leaf_documents: list[DocumentArtifact] = []
        nested_repositories: list[RepositoryArtifact] = []
        for member in members:
            member_filename = _safe_filename(member.filename, default=member.filename)
            member_route = route_attachment(filename=member_filename, mime_type="")
            if member_route.is_image:
                continue
            if member_route.category == DocumentCategory.ARCHIVE:
                if depth + 1 >= MAX_ARCHIVE_DEPTH:
                    continue
                nested_repo, nested_docs, nested_subrepos = await self._extract_archive(
                    raw=member.raw,
                    filename=member_filename,
                    mime_type=member_route.mime_type,
                    headers=headers,
                    client=client,
                    depth=depth + 1,
                )
                nested_repositories.append(nested_repo)
                nested_repositories.extend(nested_subrepos)
                leaf_documents.extend(nested_docs)
                continue

            leaf_documents.append(
                _extract_leaf_document(
                    raw=member.raw,
                    filename=member_filename,
                    mime_type=member_route.mime_type,
                    route=member_route,
                    metadata={"archive_root": filename, "archive_member": member.filename, "member_size": member.size},
                )
            )

        repository = build_repository_artifact(
            root=filename,
            documents=leaf_documents,
            metadata={
                "source": "archive",
                "mime_type": mime_type,
                "checksum": _checksum(raw),
                "member_count": len(members),
                "nested_repository_count": len(nested_repositories),
            },
        )
        return repository, leaf_documents, nested_repositories

    def _index_document(self, document: DocumentArtifact) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        if document.pages:
            for page in document.pages:
                text = page.text.strip()
                if not text:
                    continue
                chunks.append(
                    DocumentChunk(
                        document_id=document.id,
                        filename=document.filename,
                        text=text,
                        score_hint=1.5,
                        metadata={"page": page.number, **page.metadata},
                    )
                )
            return chunks

        if document.sections:
            for section in document.sections:
                text = section.text.strip()
                if not text:
                    continue
                chunks.append(
                    DocumentChunk(
                        document_id=document.id,
                        filename=document.filename,
                        text=text,
                        score_hint=1.0,
                        metadata=dict(section.metadata),
                    )
                )
            return chunks

        if document.text.strip():
            for chunk_text in _chunk_text(document.text):
                chunks.append(
                    DocumentChunk(
                        document_id=document.id,
                        filename=document.filename,
                        text=chunk_text,
                        score_hint=0.5,
                        metadata={"line_count": _line_count(chunk_text), "word_count": _word_count(chunk_text)},
                    )
                )

        return chunks

    async def process(
        self,
        *,
        attachments: list[dict[str, Any]] | None,
        session_id: str,
        headers: dict[str, str] | None = None,
    ) -> DocumentSession:
        session = self._session(session_id)
        session.last_accessed = time.time()
        self._cleanup_expired_sessions()

        if not attachments:
            return session

        seen_checksums = {artifact.checksum for artifact in session.documents if artifact.checksum}
        seen_repository_checksums = {repo.metadata.get("checksum") for repo in session.repository_artifacts if repo.metadata.get("checksum")}
        local_documents: list[DocumentArtifact] = []

        for attachment in attachments:
            if not isinstance(attachment, dict):
                continue

            try:
                filename = _safe_filename(_attachment_value(attachment, "filename", "name", "title", "path"), default="attachment")
                mime_type = _attachment_value(attachment, "mime_type", "media_type", "content_type") or ""
                route = route_attachment(filename=filename, mime_type=mime_type)

                if route.is_image:
                    continue

                raw, resolved_filename, resolved_mime = await _resolve_attachment_bytes(
                    attachment,
                    settings=self.settings,
                    headers=headers,
                    client=self.client,
                )

                if raw is None:
                    local_documents.append(
                        _failure_artifact(
                            filename=resolved_filename,
                            mime_type=resolved_mime,
                            content_type=route.category.value,
                            size=0,
                            error="attachment_unresolved",
                        )
                    )
                    continue

                checksum = _checksum(raw)
                if checksum in seen_checksums or checksum in seen_repository_checksums:
                    continue

                if route.category == DocumentCategory.ARCHIVE:
                    repository, leaf_documents, nested_repositories = await self._extract_archive(
                        raw=raw,
                        filename=resolved_filename,
                        mime_type=resolved_mime,
                        headers=headers,
                        client=self.client,
                        depth=0,
                    )
                    session.repository_artifacts.append(repository)
                    session.repository_artifacts.extend(nested_repositories)
                    local_documents.extend(leaf_documents)
                    seen_repository_checksums.add(checksum)
                    for doc in leaf_documents:
                        if doc.checksum:
                            seen_checksums.add(doc.checksum)
                    continue

                document = _extract_leaf_document(
                    raw=raw,
                    filename=resolved_filename,
                    mime_type=resolved_mime,
                    route=route,
                    metadata=attachment,
                )
                local_documents.append(document)
                seen_checksums.add(checksum)
            except Exception as exc:
                local_documents.append(
                    _failure_artifact(
                        filename=_safe_filename(_attachment_value(attachment, "filename", "name", "title", "path"), default="attachment"),
                        mime_type=_attachment_value(attachment, "mime_type", "media_type", "content_type") or "application/octet-stream",
                        content_type=route_attachment(
                            filename=_safe_filename(_attachment_value(attachment, "filename", "name", "title", "path"), default="attachment"),
                            mime_type=_attachment_value(attachment, "mime_type", "media_type", "content_type"),
                        ).category.value,
                        size=0,
                        error="document_extraction_failed",
                        exception=exc,
                    )
                )

        session.documents.extend(local_documents)
        for document in local_documents:
            session.chunks.extend(self._index_document(document))

        session.statistics = {
            "document_count": len(session.documents),
            "repository_count": len(session.repository_artifacts),
            "chunk_count": len(session.chunks),
            "supported_count": len([doc for doc in session.documents if doc.is_supported]),
            "failed_count": len([doc for doc in session.documents if not doc.is_supported]),
        }
        session.metadata.update(
            {
                "session_id": session_id,
                "attachment_count": len(attachments),
                "document_summaries": [_artifact_summary(doc) for doc in session.documents],
                "repository_summaries": [
                    {
                        "root": repo.root,
                        "statistics": repo.statistics,
                        "metadata": repo.metadata,
                        "document_count": len(repo.documents),
                    }
                    for repo in session.repository_artifacts
                ],
            }
        )
        return session

    def search(self, session_id: str, query: str, *, limit: int = 8) -> list[dict[str, Any]]:
        session = self._cache.get(session_id)
        if session is None:
            return []
        return session.search(query, limit=limit)

    def _cleanup_expired_sessions(self) -> None:
        ttl = getattr(self.settings, "document_session_ttl_s", 86400.0)
        if ttl <= 0:
            return
        now = time.time()
        expired = [session_id for session_id, session in self._cache.items() if now - session.last_accessed > ttl]
        for session_id in expired:
            self._cache.pop(session_id, None)

    def clear(self, session_id: str) -> None:
        self._cache.pop(session_id, None)

    def build_evidence(
        self,
        *,
        documents: list[DocumentArtifact],
        repository_artifacts: list[RepositoryArtifact],
        query: str | None = None,
        session_id: str | None = None,
    ) -> DocumentEvidence:
        if not documents and not repository_artifacts:
            return DocumentEvidence()

        context = ""
        excerpts: list[str] = []
        if query and session_id:
            session = self._cache.get(session_id)
            if session is not None and session.chunks:
                results = session.search(query, limit=8)
                if results:
                    lines: list[str] = []
                    for hit in results:
                        descriptor = f"File: {hit['filename']}" if hit.get("filename") else ""
                        if descriptor:
                            lines.append(descriptor)
                        excerpt_text = _truncate(str(hit.get("excerpt") or ""), 260)
                        if excerpt_text:
                            lines.append(excerpt_text)
                        lines.append("")
                    context = "\n".join(line for line in lines if line).strip()
                    excerpts = [str(hit.get("excerpt") or "") for hit in results if str(hit.get("excerpt") or "").strip()]

        if not context:
            context, excerpts = _build_document_context(documents, query=query)

        if not context and documents:
            context = "\n\n".join(
                [doc.text.strip()[:1200] for doc in documents if doc.text.strip()][:4]
            )
        summary_items: list[str] = []
        if documents:
            summary_items.append(f"{len(documents)} uploaded document(s) available.")
        if repository_artifacts:
            summary_items.append(f"{len(repository_artifacts)} repository artifact(s) available.")
        summary = " ".join(summary_items).strip() or "Uploaded documents are available."

        return DocumentEvidence(
            question=query,
            confidence=1.0 if documents or repository_artifacts else 0.0,
            summary=summary,
            context=context,
            documents=documents,
            repository_artifacts=repository_artifacts,
            excerpts=excerpts,
            metadata={
                "document_count": len(documents),
                "repository_count": len(repository_artifacts),
            },
        )
