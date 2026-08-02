from __future__ import annotations

import hashlib
import re
from typing import Any

from ..artifact import DocumentArtifact, DocumentCategory, DocumentImageReference, DocumentPage


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _line_count(text: str) -> int:
    if not text:
        return 0
    return text.count("\n") + 1


def extract_pdf_document(
    *,
    raw: bytes,
    filename: str,
    mime_type: str,
    metadata: dict[str, Any] | None = None,
) -> DocumentArtifact:
    checksum = hashlib.sha256(raw).hexdigest()
    try:
        import fitz  # type: ignore
    except Exception as exc:
        return DocumentArtifact(
            id=checksum[:16],
            filename=filename,
            extension="pdf",
            mime_type=mime_type,
            size=len(raw),
            checksum=checksum,
            category=DocumentCategory.RICH,
            content_type="pdf",
            metadata={**dict(metadata or {}), "status": "failed", "error": "pymupdf_not_available"},
            status="failed",
            error=str(exc),
        )

    try:
        doc = fitz.open(stream=raw, filetype="pdf")
        pages: list[DocumentPage] = []
        images: list[DocumentImageReference] = []
        page_texts: list[str] = []
        for index in range(doc.page_count):
            page = doc.load_page(index)
            text = page.get_text("text") or ""
            page_texts.append(text.rstrip())
            pages.append(
                DocumentPage(
                    number=index + 1,
                    text=text.rstrip(),
                    metadata={
                        "width": float(page.rect.width),
                        "height": float(page.rect.height),
                        "image_count": len(page.get_images(full=True)),
                    },
                )
            )
            for image_index, _image in enumerate(page.get_images(full=True), start=1):
                images.append(
                    DocumentImageReference(
                        reference=f"{filename}#page={index + 1}&image={image_index}",
                        description=f"Embedded image {image_index} on page {index + 1}",
                        page=index + 1,
                        metadata={"source": "pdf", "page": index + 1, "image_index": image_index},
                    )
                )

        metadata_map = dict(metadata or {})
        metadata_map.update(
            {
                "title": doc.metadata.get("title") if doc.metadata else None,
                "author": doc.metadata.get("author") if doc.metadata else None,
                "subject": doc.metadata.get("subject") if doc.metadata else None,
                "keywords": doc.metadata.get("keywords") if doc.metadata else None,
            }
        )
        text = "\n\f\n".join(page_texts).strip()
        return DocumentArtifact(
            id=checksum[:16],
            filename=filename,
            extension="pdf",
            mime_type=mime_type,
            size=len(raw),
            checksum=checksum,
            category=DocumentCategory.RICH,
            language="unknown",
            encoding="utf-8",
            content_type="pdf",
            metadata=metadata_map,
            text=text,
            pages=pages,
            images=images,
            line_count=_line_count(text),
            page_count=len(pages),
            word_count=_word_count(text),
            status="ok",
        )
    except Exception as exc:
        return DocumentArtifact(
            id=checksum[:16],
            filename=filename,
            extension="pdf",
            mime_type=mime_type,
            size=len(raw),
            checksum=checksum,
            category=DocumentCategory.RICH,
            content_type="pdf",
            metadata={**dict(metadata or {}), "status": "failed", "error": "pdf_extract_failed"},
            status="failed",
            error=str(exc),
        )

