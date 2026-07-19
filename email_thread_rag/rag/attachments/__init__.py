"""Stage 8: PDF-only attachment extraction, local OCR, and page-level chunks.

Nothing here is imported by the memory backend or the default answer path. The
Gmail extraction worker is the only production caller; heavy dependencies
(fitz/pymupdf, pytesseract) are imported lazily inside functions so this package
stays import-light for callers that only need the types.
"""
