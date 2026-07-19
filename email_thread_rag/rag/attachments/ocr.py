"""A narrow, local OCR seam.

The extractor depends only on the ``OCRBackend`` protocol -- one method that
turns a rendered page image (PNG bytes) into text. Production ships an optional
Tesseract-backed implementation behind the ``ocr`` extra; tests inject a fake.

Design rules:
  * No cloud OCR, no API key, no model download. Tesseract is a local binary.
  * When OCR is disabled or unavailable, ``build_ocr_backend`` returns ``None``.
    The extractor then records a scanned page as *unavailable* -- it never
    invents text.
"""

from __future__ import annotations

from typing import Optional, Protocol


class OCRBackend(Protocol):
    def image_to_text(self, png_bytes: bytes) -> str:
        """Return recognized text for one rendered page image (PNG bytes)."""


class TesseractOCRBackend:
    """Local Tesseract via pytesseract. Optional: imported lazily so importing
    this module never requires pytesseract/Pillow to be installed."""

    def __init__(self, *, lang: str = "eng"):
        self.lang = lang

    def image_to_text(self, png_bytes: bytes) -> str:
        import io

        import pytesseract
        from PIL import Image

        image = Image.open(io.BytesIO(png_bytes))
        return pytesseract.image_to_string(image, lang=self.lang).strip()


def build_ocr_backend(settings) -> Optional[OCRBackend]:
    """Return an OCR backend, or ``None`` when OCR is disabled or unavailable.

    ``None`` is a safe, expected state: the extractor records scanned pages as
    unavailable instead of guessing at their contents.
    """
    if not getattr(settings, "attachment_ocr_enabled", False):
        return None
    try:  # pragma: no cover - depends on the optional 'ocr' extra being installed
        import pytesseract  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError:
        return None
    return TesseractOCRBackend()
