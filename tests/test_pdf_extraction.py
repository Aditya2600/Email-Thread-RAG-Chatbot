"""Stage-8 PDF-only extraction: native text, fake-OCR fallback, safe rejection.

No real OCR engine is used -- a fake backend stands in. Production code must work
when OCR is disabled/unavailable and record a scanned page as unavailable rather
than invent text.
"""

from __future__ import annotations

import fitz
import pytest
from PIL import Image, ImageDraw

from email_thread_rag.config import Settings
from email_thread_rag.rag.attachments.pdf import PDF_MEDIA_TYPE, extract_pdf


class FakeOCRBackend:
    def __init__(self, text: str):
        self.text = text
        self.calls = 0

    def image_to_text(self, png_bytes: bytes) -> str:
        self.calls += 1
        return self.text


def _settings(**overrides) -> Settings:
    return Settings(**overrides)


def _text_pdf(pages: list[str]) -> bytes:
    document = fitz.open()
    for body in pages:
        page = document.new_page(width=400, height=400)
        page.insert_text((40, 60), body, fontsize=14)
    return document.tobytes()


def _image_only_pdf(text_in_image: str) -> bytes:
    image = Image.new("RGB", (500, 200), color="white")
    ImageDraw.Draw(image).text((20, 80), text_in_image, fill="black")
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    document = fitz.open()
    page = document.new_page(width=500, height=200)
    page.insert_image(page.rect, stream=buffer.getvalue())
    return document.tobytes()


def _extract(data, *, settings=None, ocr_backend=None, filename="doc.pdf", media_type=PDF_MEDIA_TYPE):
    return extract_pdf(
        data,
        attachment_id="att-1",
        filename=filename,
        message_id="<m1@x>",
        thread_id="thread-1",
        media_type=media_type,
        settings=settings or _settings(),
        ocr_backend=ocr_backend,
    )


def test_multi_page_text_pdf_yields_native_pages():
    result = _extract(_text_pdf(["First page approved amount $1200.", "Second page signed by Bob."]))
    assert result.status == "done"
    assert result.method == "native_pdf"
    assert [p.page_no for p in result.record.pages] == [1, 2]
    assert "1200" in result.record.pages[0].text
    assert all(p.ocr_used is False for p in result.record.pages)


def test_scanned_page_uses_fake_ocr_fallback():
    ocr = FakeOCRBackend("Scanned approval amount $2222")
    result = _extract(_image_only_pdf("Scanned approval amount $2222"), ocr_backend=ocr)
    assert result.status == "done"
    assert result.method == "ocr"
    assert ocr.calls == 1
    page = result.record.pages[0]
    assert page.ocr_used is True
    assert "$2222" in page.text


def test_ocr_disabled_records_page_as_unavailable_without_inventing_text():
    result = _extract(_image_only_pdf("Nobody will read this"), ocr_backend=None)
    assert result.status == "done"
    assert result.method == "native_pdf"  # no OCR happened
    page = result.record.pages[0]
    assert page.ocr_used is False
    assert page.text == ""  # unavailable, not invented


def test_unsupported_media_type_is_rejected():
    result = _extract(b"hello", filename="note.txt", media_type="text/plain")
    assert result.status == "unsupported"
    assert result.error == "unsupported_media_type"
    assert result.record is None


def test_oversized_pdf_is_rejected_safely():
    result = _extract(_text_pdf(["x" * 100]), settings=_settings(attachment_max_bytes=10))
    assert result.status == "failed"
    assert result.error == "oversized"
    assert result.record is None


def test_malformed_pdf_is_rejected_safely():
    result = _extract(b"%PDF-1.4 this is not really a pdf")
    assert result.status == "failed"
    assert result.error == "malformed"
    assert result.record is None


def test_encrypted_pdf_is_rejected_safely(tmp_path):
    document = fitz.open()
    page = document.new_page()
    page.insert_text((40, 60), "secret budget", fontsize=14)
    path = tmp_path / "enc.pdf"
    document.save(
        str(path),
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="owner",
        user_pw="user",
    )
    result = _extract(path.read_bytes())
    assert result.status == "failed"
    assert result.error == "encrypted"
    assert result.record is None
