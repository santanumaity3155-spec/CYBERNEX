"""Standalone synthetic OCR smoke test (Phase 8).

Verifies local PaddleOCR text recognition on a purely synthetic
industrial inspection document image and scanned PDF. No Aadhaar or
personal data is used.
"""

import io
import os
import time
import tempfile
import fitz
import pytest
from PIL import Image, ImageDraw, ImageFont

from app.services.ocr.service import ocr_service, PADDLEOCR_AVAILABLE
from app.tools.pdf_tool import extract_pdf_text

requires_ocr = pytest.mark.skipif(
    not ocr_service.is_available,
    reason="Local PaddleOCR engine is unavailable in this test environment",
)

SYNTHETIC_INSPECTION_LINES = [
    "CYBERNEX OCR TEST",
    "Industrial Safety Inspection Report",
    "Equipment: Pump-204",
    "Pressure: 10 bar",
    "Temperature: 85 C",
    "Status: NORMAL",
]


def _build_synthetic_inspection_image_png() -> bytes:
    """Generate a clean synthetic industrial inspection image in PNG bytes."""
    width = 1400
    height = 140 + 130 * len(SYNTHETIC_INSPECTION_LINES)
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=56)
    except TypeError:
        font = ImageFont.load_default()

    y = 60
    for line in SYNTHETIC_INSPECTION_LINES:
        draw.text((80, y), line, fill="black", font=font)
        y += 130

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _build_synthetic_scanned_pdf_bytes() -> bytes:
    """Wrap synthetic inspection image as an image-only (scanned) PDF page."""
    png_bytes = _build_synthetic_inspection_image_png()
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4
    page.insert_image(fitz.Rect(30, 30, 565, 812), stream=png_bytes, keep_proportion=True)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@requires_ocr
def test_synthetic_ocr_image_smoke():
    """Smoke test: ocr_service.ocr_image extracts industrial inspection fields from synthetic image."""
    png_bytes = _build_synthetic_inspection_image_png()
    assert len(png_bytes) > 0

    t_start = time.perf_counter()
    extracted = ocr_service.ocr_image(png_bytes)
    duration = time.perf_counter() - t_start

    print(f"\n[Smoke Test] Image OCR completed in {duration:.2f}s")
    print(f"[Smoke Test] Extracted text:\n{extracted}")

    # Must recognize non-empty text
    assert extracted.strip(), "OCR must return non-empty text"

    # Verify key tokens
    assert "CYBERNEX" in extracted
    assert any(tok in extracted for tok in ("Pump-204", "Pump", "204"))
    assert any(tok in extracted for tok in ("NORMAL", "NORM"))
    assert any(tok in extracted for tok in ("Pressure", "10 bar", "bar"))
    assert any(tok in extracted for tok in ("Temperature", "85 C", "85"))


@requires_ocr
def test_synthetic_scanned_pdf_smoke(tmp_path):
    """Smoke test: extract_pdf_text extracts page-level text from synthetic scanned PDF."""
    pdf_bytes = _build_synthetic_scanned_pdf_bytes()
    pdf_file = tmp_path / "synthetic_smoke.pdf"
    pdf_file.write_bytes(pdf_bytes)

    t_start = time.perf_counter()
    result = extract_pdf_text(str(pdf_file), use_ocr=True)
    duration = time.perf_counter() - t_start

    print(f"\n[Smoke Test] Scanned PDF OCR completed in {duration:.2f}s")
    print(f"[Smoke Test] Result: {result}")

    assert result["page_count"] == 1
    page = result["pages"][0]
    assert page["page_number"] == 1
    assert page["source"] == "ocr"
    assert page["has_text"] is True
    assert page["character_count"] > 0

    page_text = page["text"]
    assert "CYBERNEX" in page_text
    assert any(tok in page_text for tok in ("Pump", "204"))
    assert any(tok in page_text for tok in ("NORMAL", "NORM"))
    assert any(tok in page_text for tok in ("10 bar", "bar", "Pressure"))
    assert any(tok in page_text for tok in ("85", "Temperature"))
