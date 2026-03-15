from __future__ import annotations

import io
import mimetypes
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import fitz
import pytesseract
from bs4 import BeautifulSoup
from docx import Document
from PIL import Image

try:
    from striprtf.striprtf import rtf_to_text as _rtf_to_text
except ImportError:
    _rtf_to_text = None

try:
    import xlrd as _xlrd
except ImportError:
    _xlrd = SimpleNamespace(open_workbook=None)

xlrd = _xlrd

from email_thread_rag.app.schemas import AttachmentPage, AttachmentRecord
from email_thread_rag.config import Settings


def detect_media_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def _page_density(page: fitz.Page, text: str) -> tuple[int, float]:
    alnum_count = len(re.findall(r"[A-Za-z0-9]", text))
    area = max(page.rect.width * page.rect.height, 1.0)
    density = alnum_count / area
    return alnum_count, density


def _ocr_page(page: fitz.Page) -> str:
    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    image = Image.open(io.BytesIO(pixmap.tobytes("png")))
    return pytesseract.image_to_string(image)


def parse_pdf_attachment(
    path: Path,
    *,
    attachment_id: str,
    message_id: str,
    thread_id: str,
    settings: Settings,
) -> AttachmentRecord:
    document = fitz.open(path)
    pages: list[AttachmentPage] = []
    for index, page in enumerate(document, start=1):
        extracted = page.get_text("text") or ""
        alnum_count, density = _page_density(page, extracted)
        ocr_used = False
        if (
            alnum_count < settings.ocr_thresholds.min_alnum_chars
            or density < settings.ocr_thresholds.min_text_density
        ):
            extracted = _ocr_page(page).strip()
            alnum_count, density = _page_density(page, extracted)
            ocr_used = True
        pages.append(
            AttachmentPage(
                page_no=index,
                text=extracted.strip(),
                ocr_used=ocr_used,
                text_density=density,
                alnum_count=alnum_count,
            )
        )
    return AttachmentRecord(
        attachment_id=attachment_id,
        message_id=message_id,
        thread_id=thread_id,
        filename=path.name,
        media_type="application/pdf",
        source_path=str(path),
        pages=pages,
    )


def parse_docx_attachment(path: Path, *, attachment_id: str, message_id: str, thread_id: str) -> AttachmentRecord:
    document = Document(path)
    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
    text = "\n".join(paragraphs)
    return AttachmentRecord(
        attachment_id=attachment_id,
        message_id=message_id,
        thread_id=thread_id,
        filename=path.name,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        source_path=str(path),
        pages=[AttachmentPage(page_no=1, text=text, ocr_used=False, text_density=float(len(text)), alnum_count=len(text))],
    )


def _make_single_page_record(
    *,
    path: Path,
    attachment_id: str,
    message_id: str,
    thread_id: str,
    media_type: str,
    text: str,
) -> AttachmentRecord:
    normalized_text = text.strip()
    return AttachmentRecord(
        attachment_id=attachment_id,
        message_id=message_id,
        thread_id=thread_id,
        filename=path.name,
        media_type=media_type,
        source_path=str(path),
        pages=[
            AttachmentPage(
                page_no=1,
                text=normalized_text,
                ocr_used=False,
                text_density=float(len(normalized_text)),
                alnum_count=len(normalized_text),
            )
        ],
    )


def parse_doc_attachment(path: Path, *, attachment_id: str, message_id: str, thread_id: str) -> AttachmentRecord:
    text = ""
    antiword = shutil.which("antiword")
    if antiword:
        result = subprocess.run([antiword, str(path)], capture_output=True, text=True, check=False)
        text = (result.stdout or "").strip()
    elif shutil.which("textutil"):
        result = subprocess.run(
            ["textutil", "-convert", "txt", "-stdout", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        text = (result.stdout or "").strip()
    if not text:
        text = path.read_text(encoding="utf-8", errors="ignore")
    return _make_single_page_record(
        path=path,
        attachment_id=attachment_id,
        message_id=message_id,
        thread_id=thread_id,
        media_type="application/msword",
        text=text,
    )


def parse_rtf_attachment(path: Path, *, attachment_id: str, message_id: str, thread_id: str) -> AttachmentRecord:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if _rtf_to_text is not None:
        text = _rtf_to_text(raw)
    else:
        text = re.sub(r"\\[a-zA-Z0-9]+ ?", "", raw)
        text = text.replace("{", "").replace("}", "")
    return _make_single_page_record(
        path=path,
        attachment_id=attachment_id,
        message_id=message_id,
        thread_id=thread_id,
        media_type="application/rtf",
        text=text,
    )


def parse_xls_attachment(path: Path, *, attachment_id: str, message_id: str, thread_id: str) -> AttachmentRecord:
    if not getattr(xlrd, "open_workbook", None):
        return parse_text_attachment(path, attachment_id=attachment_id, message_id=message_id, thread_id=thread_id)
    workbook = xlrd.open_workbook(path, on_demand=True)
    parts: list[str] = []
    for sheet in workbook.sheets():
        parts.append(f"[Sheet: {sheet.name}]")
        for row_idx in range(sheet.nrows):
            values = [str(sheet.cell_value(row_idx, col_idx)).strip() for col_idx in range(sheet.ncols)]
            values = [value for value in values if value]
            if values:
                parts.append(" | ".join(values))
    workbook.release_resources()
    text = "\n".join(parts)
    return _make_single_page_record(
        path=path,
        attachment_id=attachment_id,
        message_id=message_id,
        thread_id=thread_id,
        media_type="application/vnd.ms-excel",
        text=text,
    )


def parse_text_attachment(path: Path, *, attachment_id: str, message_id: str, thread_id: str) -> AttachmentRecord:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    return _make_single_page_record(
        path=path,
        attachment_id=attachment_id,
        message_id=message_id,
        thread_id=thread_id,
        media_type=detect_media_type(path),
        text=text,
    )


def parse_html_attachment(path: Path, *, attachment_id: str, message_id: str, thread_id: str) -> AttachmentRecord:
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = "\n".join(chunk.strip() for chunk in soup.stripped_strings if chunk.strip())
    return _make_single_page_record(
        path=path,
        attachment_id=attachment_id,
        message_id=message_id,
        thread_id=thread_id,
        media_type="text/html",
        text=text,
    )


def parse_attachment(
    path: Path,
    *,
    attachment_id: str,
    message_id: str,
    thread_id: str,
    settings: Settings,
    media_type: Optional[str] = None,
) -> AttachmentRecord:
    suffix = path.suffix.lower()
    effective_media_type = media_type or detect_media_type(path)
    if suffix == ".pdf" or effective_media_type == "application/pdf":
        return parse_pdf_attachment(
            path,
            attachment_id=attachment_id,
            message_id=message_id,
            thread_id=thread_id,
            settings=settings,
        )
    if suffix == ".docx":
        return parse_docx_attachment(path, attachment_id=attachment_id, message_id=message_id, thread_id=thread_id)
    if suffix == ".doc" or effective_media_type == "application/msword":
        return parse_doc_attachment(path, attachment_id=attachment_id, message_id=message_id, thread_id=thread_id)
    if suffix == ".xls":
        return parse_xls_attachment(path, attachment_id=attachment_id, message_id=message_id, thread_id=thread_id)
    if suffix == ".rtf":
        return parse_rtf_attachment(path, attachment_id=attachment_id, message_id=message_id, thread_id=thread_id)
    if suffix in {".txt", ".text"} or effective_media_type.startswith("text/plain"):
        return parse_text_attachment(path, attachment_id=attachment_id, message_id=message_id, thread_id=thread_id)
    if suffix in {".html", ".htm"} or effective_media_type == "text/html":
        return parse_html_attachment(path, attachment_id=attachment_id, message_id=message_id, thread_id=thread_id)
    return parse_text_attachment(path, attachment_id=attachment_id, message_id=message_id, thread_id=thread_id)
