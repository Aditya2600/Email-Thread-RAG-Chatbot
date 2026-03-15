from __future__ import annotations

from pathlib import Path

import fitz
from PIL import Image, ImageDraw

from email_thread_rag.rag.parse_attachments import parse_pdf_attachment


def test_ocr_fallback_on_image_only_pdf(monkeypatch, tmp_path, sample_records):
    settings, _, _, _ = sample_records
    image_path = tmp_path / "page.png"
    pdf_path = tmp_path / "image_only.pdf"

    image = Image.new("RGB", (500, 200), color="white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 80), "Scanned approval amount $2222", fill="black")
    image.save(image_path)

    document = fitz.open()
    page = document.new_page(width=500, height=200)
    page.insert_image(page.rect, filename=str(image_path))
    document.save(pdf_path)

    monkeypatch.setattr("email_thread_rag.rag.parse_attachments._ocr_page", lambda page: "Scanned approval amount $2222")
    record = parse_pdf_attachment(
        pdf_path,
        attachment_id="ocr-att-1",
        message_id="<ocr@example.com>",
        thread_id="thread-ocr",
        settings=settings,
    )
    assert record.pages[0].ocr_used is True
    assert record.pages[0].page_no == 1
    assert "$2222" in record.pages[0].text

