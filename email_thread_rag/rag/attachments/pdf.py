"""PDF-only extraction: native text first, local OCR fallback, safe rejection.

Flow per attachment:
    decoded bytes -> open PDF -> per page: native text extraction; if a page has
    no usable native text, OCR it *only if* an OCR backend is available, else
    record the page as unavailable (never invent text).

Only ``application/pdf`` is handled. Unsupported, oversized, encrypted,
password-protected, or malformed inputs fail safely: ``ExtractionResult`` comes
back with ``status='failed'`` (or ``'unsupported'``) and no record, so nothing
enters retrieval.

``fitz`` (pymupdf) is imported lazily so importing this module is cheap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from email_thread_rag.app.schemas import AttachmentPage, AttachmentRecord
from email_thread_rag.rag.attachments.ocr import OCRBackend

PDF_MEDIA_TYPE = "application/pdf"


@dataclass
class ExtractionResult:
    """Outcome of one attachment extraction attempt.

    ``record`` is a page-segmented ``AttachmentRecord`` on success, else ``None``.
    ``method`` is ``native_pdf`` | ``ocr`` | ``mixed`` | ``None``. ``error`` is a
    safe, loggable rule name (``oversized`` / ``encrypted`` / ``malformed`` /
    ``unsupported_media_type``), never document bytes.
    """

    status: str  # done | failed | unsupported
    record: Optional[AttachmentRecord] = None
    method: Optional[str] = None
    error: Optional[str] = None


def _alnum_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]", text))


def _page_is_usable(text: str, thresholds) -> bool:
    # "Usable native text" is really "did this page extract as text or is it an
    # image-only/scanned page?". The reliable signal is the count of alphanumeric
    # characters the native extractor recovered; a scanned page yields ~none.
    return _alnum_count(text) >= thresholds.min_alnum_chars


def extract_pdf(
    data: bytes,
    *,
    attachment_id: str,
    filename: str,
    message_id: str,
    thread_id: str,
    media_type: str = PDF_MEDIA_TYPE,
    settings,
    ocr_backend: Optional[OCRBackend] = None,
) -> ExtractionResult:
    """Extract page text from PDF bytes. Never raises on bad input -- it returns
    a failed/unsupported ``ExtractionResult`` instead."""
    if media_type != PDF_MEDIA_TYPE and not filename.lower().endswith(".pdf"):
        return ExtractionResult(status="unsupported", error="unsupported_media_type")

    max_bytes = int(getattr(settings, "attachment_max_bytes", 20_000_000))
    if len(data) > max_bytes:
        return ExtractionResult(status="failed", error="oversized")

    import fitz  # lazy: pymupdf is only needed inside the extraction worker

    try:
        document = fitz.open(stream=data, filetype="pdf")
    except Exception:  # noqa: BLE001 - any parse failure is a safe rejection
        return ExtractionResult(status="failed", error="malformed")

    # Encrypted / password-protected: needs_pass is set until authenticated.
    if getattr(document, "needs_pass", False) or getattr(document, "is_encrypted", False):
        document.close()
        return ExtractionResult(status="failed", error="encrypted")

    thresholds = settings.ocr_thresholds
    pages: list[AttachmentPage] = []
    saw_native = False
    saw_ocr = False
    try:
        for index, page in enumerate(document, start=1):
            native = (page.get_text("text") or "").strip()
            if _page_is_usable(native, thresholds):
                pages.append(_page(index, native, ocr_used=False))
                saw_native = True
                continue
            # No usable native text: OCR only if a backend is available.
            if ocr_backend is not None:
                png = page.get_pixmap(matrix=fitz.Matrix(2, 2)).tobytes("png")
                ocr_text = (ocr_backend.image_to_text(png) or "").strip()
                pages.append(_page(index, ocr_text, ocr_used=True))
                saw_ocr = saw_ocr or bool(ocr_text)
            else:
                # Unavailable: record the page, but with no invented text. It
                # produces no chunk, so it never reaches retrieval.
                pages.append(_page(index, "", ocr_used=False))
    except Exception:  # noqa: BLE001 - a mid-document decode error is a safe reject
        document.close()
        return ExtractionResult(status="failed", error="malformed")
    finally:
        document.close()

    method = "mixed" if (saw_native and saw_ocr) else ("ocr" if saw_ocr else "native_pdf")
    record = AttachmentRecord(
        attachment_id=attachment_id,
        message_id=message_id,
        thread_id=thread_id,
        filename=filename,
        media_type=PDF_MEDIA_TYPE,
        source_path=f"attachment://{message_id}/{attachment_id}",
        pages=pages,
    )
    return ExtractionResult(status="done", record=record, method=method)


def _page(page_no: int, text: str, *, ocr_used: bool) -> AttachmentPage:
    alnum = _alnum_count(text)
    return AttachmentPage(
        page_no=page_no,
        text=text,
        ocr_used=ocr_used,
        text_density=float(alnum),
        alnum_count=alnum,
    )
