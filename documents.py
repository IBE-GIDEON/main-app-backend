from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import PyPDF2
from fastapi import UploadFile
from pydantic import BaseModel, Field

from router import UserContext

logger = logging.getLogger("three-ai.documents")

_MAX_COMBINED_CHARS = int(os.getenv("THINK_AI_DOCUMENT_MAX_CHARS", "120000"))
_STORE_VERSION = "1.0.0"
_STORE_ROOT = Path(__file__).resolve().parent / "manual_data_store"

_write_locks: dict[str, asyncio.Lock] = {}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n\n[Document context truncated for speed.]"


def _get_lock(company_id: str) -> asyncio.Lock:
    if company_id not in _write_locks:
        _write_locks[company_id] = asyncio.Lock()
    return _write_locks[company_id]


def _company_hash(company_id: str) -> str:
    return hashlib.sha256(company_id.encode("utf-8")).hexdigest()


def _company_dir(company_id: str) -> Path:
    return _STORE_ROOT / _company_hash(company_id)


def _assets_dir(company_id: str) -> Path:
    return _company_dir(company_id) / "assets"


def _texts_dir(company_id: str) -> Path:
    return _company_dir(company_id) / "texts"


def _manifest_path(company_id: str) -> Path:
    return _company_dir(company_id) / "manifest.json"


def _ensure_company_dirs(company_id: str) -> None:
    _assets_dir(company_id).mkdir(parents=True, exist_ok=True)
    _texts_dir(company_id).mkdir(parents=True, exist_ok=True)


def _safe_filename(filename: str) -> str:
    base = Path(filename or "uploaded-file").name.strip() or "uploaded-file"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", base).strip(".-_")
    return safe or "uploaded-file"


def _normalize_preview(text: str, max_chars: int = 220) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "…"


def _extract_pdf_text(content: bytes) -> tuple[str, int]:
    reader = PyPDF2.PdfReader(io.BytesIO(content))
    pages = 0
    parts: list[str] = []

    for page in reader.pages:
        pages += 1
        text = page.extract_text()
        if text:
            parts.append(text.strip())

    return "\n\n".join(part for part in parts if part), pages


def _extract_text(content: bytes, filename: str) -> tuple[str, int]:
    lowered = filename.lower()

    if lowered.endswith(".pdf"):
        try:
            return _extract_pdf_text(content)
        except Exception as exc:
            logger.warning("PDF extraction failed | file=%s | %s", filename, exc)
            return "", 0

    if lowered.endswith((".csv", ".txt", ".md", ".json")):
        return content.decode("utf-8", errors="ignore"), 1

    return content.decode("utf-8", errors="ignore"), 1


class UploadedDocumentMeta(BaseModel):
    document_id: str
    filename: str
    pages: int = 0
    chars_extracted: int = 0
    uploaded_utc: str = Field(default_factory=_utc_now_iso)
    sha256: str
    size_bytes: int = 0
    content_type: str | None = None
    preview_text: str = ""


class StoredDocumentRecord(UploadedDocumentMeta):
    stored_filename: str
    text_filename: str


class CompanyDocumentStore(BaseModel):
    company_id: str
    version: str = _STORE_VERSION
    created_utc: str = Field(default_factory=_utc_now_iso)
    last_updated_utc: str = Field(default_factory=_utc_now_iso)
    documents: list[StoredDocumentRecord] = Field(default_factory=list)


class StoredDocumentBundle(BaseModel):
    company_id: str
    combined_text: str
    uploaded_utc: str
    total_chars: int
    document_count: int
    documents: list[UploadedDocumentMeta] = Field(default_factory=list)


class UploadedDocumentDetail(BaseModel):
    document: UploadedDocumentMeta
    extracted_text: str


def _new_store(company_id: str) -> CompanyDocumentStore:
    now = _utc_now_iso()
    return CompanyDocumentStore(
        company_id=company_id,
        created_utc=now,
        last_updated_utc=now,
        documents=[],
    )


def _read_store_sync(company_id: str) -> CompanyDocumentStore:
    path = _manifest_path(company_id)
    if not path.exists():
        return _new_store(company_id)

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        store = CompanyDocumentStore.model_validate(raw)
        if not store.company_id:
            store.company_id = company_id
        return store
    except Exception as exc:
        logger.error("Document manifest parse error | company=%s | %s", company_id, exc)
        return _new_store(company_id)


def _write_store_sync(company_id: str, store: CompanyDocumentStore) -> None:
    _ensure_company_dirs(company_id)
    store.last_updated_utc = _utc_now_iso()
    _manifest_path(company_id).write_text(
        json.dumps(store.model_dump(), indent=2),
        encoding="utf-8",
    )


def _record_asset_path(company_id: str, record: StoredDocumentRecord) -> Path:
    return _assets_dir(company_id) / record.stored_filename


def _record_text_path(company_id: str, record: StoredDocumentRecord) -> Path:
    return _texts_dir(company_id) / record.text_filename


def _load_record_text_sync(company_id: str, record: StoredDocumentRecord) -> str:
    path = _record_text_path(company_id, record)
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    except Exception as exc:
        logger.error(
            "Document text read failed | company=%s | document_id=%s | %s",
            company_id,
            record.document_id,
            exc,
        )
        return ""


def _meta_from_record(record: StoredDocumentRecord) -> UploadedDocumentMeta:
    return UploadedDocumentMeta(
        document_id=record.document_id,
        filename=record.filename,
        pages=record.pages,
        chars_extracted=record.chars_extracted,
        uploaded_utc=record.uploaded_utc,
        sha256=record.sha256,
        size_bytes=record.size_bytes,
        content_type=record.content_type,
        preview_text=record.preview_text,
    )


def _build_bundle_sync(company_id: str, store: CompanyDocumentStore) -> StoredDocumentBundle:
    ordered_documents = sorted(
        store.documents,
        key=lambda item: item.uploaded_utc,
        reverse=True,
    )
    sections: list[str] = [f"--- UPLOADED FINANCIAL DOCUMENTATION FOR {company_id} ---"]
    documents: list[UploadedDocumentMeta] = []

    for record in ordered_documents:
        documents.append(_meta_from_record(record))
        extracted_text = _load_record_text_sync(company_id, record).strip()
        rendered_text = (
            extracted_text
            if extracted_text
            else f"[{record.filename} was uploaded, but no readable text could be extracted.]"
        )
        sections.append(f"Document Name: {record.filename}")
        sections.append(rendered_text)
        sections.append("------------------------")

    combined_text = _truncate("\n\n".join(section for section in sections if section), _MAX_COMBINED_CHARS)

    return StoredDocumentBundle(
        company_id=company_id,
        combined_text=combined_text if documents else "",
        uploaded_utc=store.last_updated_utc,
        total_chars=len(combined_text) if documents else 0,
        document_count=len(documents),
        documents=documents,
    )


async def store_uploaded_documents(company_id: str, files: list[UploadFile]) -> StoredDocumentBundle:
    lock = _get_lock(company_id)

    async with lock:
        store = _read_store_sync(company_id)
        _ensure_company_dirs(company_id)

        for file in files:
            content = await file.read()
            extracted_text, pages = _extract_text(content, file.filename or "uploaded-file")
            normalized_text = extracted_text.strip()
            stored_text = normalized_text or f"[No readable text could be extracted from {file.filename or 'uploaded-file'}.]"
            file_hash = hashlib.sha256(content).hexdigest()
            uploaded_utc = _utc_now_iso()
            safe_name = _safe_filename(file.filename or "uploaded-file")
            document_id = hashlib.sha256(
                f"{company_id}:{safe_name}:{file_hash}:{uploaded_utc}".encode("utf-8")
            ).hexdigest()[:16]
            stored_filename = f"{document_id}_{safe_name}"
            text_filename = f"{document_id}.txt"

            (_assets_dir(company_id) / stored_filename).write_bytes(content)
            (_texts_dir(company_id) / text_filename).write_text(stored_text, encoding="utf-8")

            store.documents.append(
                StoredDocumentRecord(
                    document_id=document_id,
                    filename=file.filename or "uploaded-file",
                    pages=pages,
                    chars_extracted=len(normalized_text),
                    uploaded_utc=uploaded_utc,
                    sha256=file_hash,
                    size_bytes=len(content),
                    content_type=file.content_type,
                    preview_text=_normalize_preview(stored_text),
                    stored_filename=stored_filename,
                    text_filename=text_filename,
                )
            )

        _write_store_sync(company_id, store)
        bundle = _build_bundle_sync(company_id, store)

    logger.info(
        "Stored uploaded documents | company=%s | count=%d | chars=%d",
        company_id,
        bundle.document_count,
        bundle.total_chars,
    )
    return bundle


def get_uploaded_documents(company_id: str) -> StoredDocumentBundle:
    return _build_bundle_sync(company_id, _read_store_sync(company_id))


def get_uploaded_document_detail(
    company_id: str,
    document_id: str,
) -> UploadedDocumentDetail | None:
    store = _read_store_sync(company_id)
    record = next((item for item in store.documents if item.document_id == document_id), None)
    if record is None:
        return None

    return UploadedDocumentDetail(
        document=_meta_from_record(record),
        extracted_text=_load_record_text_sync(company_id, record),
    )


def get_uploaded_document_asset(
    company_id: str,
    document_id: str,
) -> tuple[Path, UploadedDocumentMeta] | None:
    store = _read_store_sync(company_id)
    record = next((item for item in store.documents if item.document_id == document_id), None)
    if record is None:
        return None

    asset_path = _record_asset_path(company_id, record)
    if not asset_path.exists():
        return None

    return asset_path, _meta_from_record(record)


async def delete_uploaded_document(company_id: str, document_id: str) -> StoredDocumentBundle:
    lock = _get_lock(company_id)

    async with lock:
        store = _read_store_sync(company_id)
        target = next((item for item in store.documents if item.document_id == document_id), None)
        if target is None:
            raise KeyError(document_id)

        store.documents = [item for item in store.documents if item.document_id != document_id]

        for path in (_record_asset_path(company_id, target), _record_text_path(company_id, target)):
            try:
                path.unlink(missing_ok=True)
            except Exception as exc:
                logger.warning(
                    "Failed to remove document artifact | company=%s | document_id=%s | %s",
                    company_id,
                    document_id,
                    exc,
                )

        _write_store_sync(company_id, store)
        bundle = _build_bundle_sync(company_id, store)

    logger.info(
        "Deleted uploaded document | company=%s | document_id=%s | remaining=%d",
        company_id,
        document_id,
        bundle.document_count,
    )
    return bundle


async def inject_documents_into_context(company_id: str, ctx: UserContext | None) -> UserContext:
    bundle = get_uploaded_documents(company_id)

    if ctx is None:
        ctx = UserContext(extra={})
    if ctx.extra is None:
        ctx.extra = {}

    if bundle.document_count == 0:
        return ctx

    ctx.extra["Uploaded Financial Documents"] = bundle.combined_text
    ctx.extra["uploaded_document_count"] = bundle.document_count
    ctx.extra["uploaded_document_names"] = [doc.filename for doc in bundle.documents]
    ctx.extra["uploaded_document_updated_utc"] = bundle.uploaded_utc

    logger.info(
        "Injected uploaded documents into context | company=%s | count=%d",
        company_id,
        bundle.document_count,
    )
    return ctx


def build_document_bundle_response(bundle: StoredDocumentBundle) -> dict[str, Any]:
    return {
        "ok": True,
        "document_count": bundle.document_count,
        "total_chars": bundle.total_chars,
        "uploaded_utc": bundle.uploaded_utc,
        "documents": [doc.model_dump() for doc in bundle.documents],
    }


def build_document_upload_response(bundle: StoredDocumentBundle) -> dict[str, Any]:
    return {
        **build_document_bundle_response(bundle),
        "message": f"Manual data deck updated with {bundle.document_count} stored document(s).",
    }


def build_document_detail_response(detail: UploadedDocumentDetail) -> dict[str, Any]:
    return {
        "ok": True,
        "document": detail.document.model_dump(),
        "extracted_text": detail.extracted_text,
    }
