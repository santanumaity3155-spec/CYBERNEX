"""
PDF processing tool (Phase 7 + Phase 8 OCR integration).

Local, page-level text extraction for PDFs using PyMuPDF (fitz), with an
optional PaddleOCR fallback for scanned/image-only pages:

                    PDF
                     |
                     v
                PDF Processor
                     |
            Can text be extracted?
               /            \
             YES              NO
              |               |
              v               v
          PyMuPDF         Render page
                             |
                             v
                         PaddleOCR
                             |
             (clean text merged back per-page)

Sovereignty guarantees:
- Runs 100% locally. No cloud APIs, no external services, no network calls.
- When ``use_ocr=True``, pages without meaningful extractable text are
  rendered in memory (PNG bytes) and transcribed page-by-page with the local
  PaddleOCR engine. No rendered images are stored on disk.

Output contract (consumed by the API layer; OCR/RAG/LangGraph
phases consume the same plain-dict structure directly):

{
    "filename": "inspection_report.pdf",
    "page_count": 3,
    "pages": [
        {"page_number": 1, "text": "...", "character_count": 512, "has_text": True, "source": "pymupdf"},
        {"page_number": 2, "text": "...", "character_count": 87,  "has_text": True, "source": "ocr"},
        {"page_number": 3, "text": "",    "character_count": 0,   "has_text": False, "source": "ocr"}
    ]
}

The per-page ``source`` field is metadata: "pymupdf" (native text layer),
"ocr" (PaddleOCR output). It is informational only and is not part of the
public API response schema.
"""

import concurrent.futures
import os
import time
from typing import Any, Dict, List

import fitz  # PyMuPDF

from app.core.logging import logger

SOURCE_PYMUPDF = "pymupdf"
SOURCE_OCR = "ocr"

# Pages render at 200 DPI for OCR: high enough for clear text recognition,
# small enough to keep images memory-friendly.
OCR_RENDER_DPI = 200

# Per-page OCR timeout to prevent a single complex page from stalling requests.
PAGE_OCR_TIMEOUT_SECONDS = 60.0


class PDFExtractionError(Exception):
    """Raised when a PDF cannot be opened, decrypted, or processed locally.

    The message is intentionally generic and safe to expose to API clients;
    internal details are only written to the local log.
    """


def clean_page_text(raw_text: str) -> str:
    """Lightweight whitespace normalization for an extracted page.

    Preserves document structure (headings, paragraphs, numbers, units,
    tables/line layout as provided by PyMuPDF) while removing obvious noise:
    - normalizes CRLF / CR line endings to LF
    - normalizes non-breaking spaces to regular spaces
    - strips trailing whitespace per line (leading indentation is kept so
      table alignment survives)
    - collapses repeated blank lines to a single blank line
    - drops leading/trailing blank lines
    """
    if not raw_text:
        return ""

    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")

    lines = [line.rstrip() for line in text.split("\n")]

    cleaned: List[str] = []
    blank_run = 0
    for line in lines:
        if not line:
            blank_run += 1
            if blank_run == 1:
                cleaned.append(line)
        else:
            blank_run = 0
            cleaned.append(line)

    while cleaned and not cleaned[0]:
        cleaned.pop(0)
    while cleaned and not cleaned[-1]:
        cleaned.pop()

    return "\n".join(cleaned)


def _render_page_to_png_bytes(page: Any, dpi: int = OCR_RENDER_DPI) -> bytes:
    """Render a PDF page to in-memory PNG bytes at the given resolution.

    No image file is written to disk; the raw PNG payload is returned so the
    OCR service can decode it in memory.
    """
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return pix.tobytes("png")


def _extract_page_text_via_ocr(page: Any, page_number: int, dpi: int) -> str:
    """Render a page and transcribe it with the local PaddleOCR engine.

    Raises:
        PDFExtractionError: If the local OCR engine is unavailable or fails.
    """
    # Deferred import keeps pdf_tool lightweight and avoids a hard dependency
    # on PaddleOCR for purely text-based workflows.
    from app.services.ocr.service import (
        OCRProcessingError,
        OCRUnavailableError,
        ocr_service,
    )

    logger.info(f"Page {page_number} requires OCR.")

    if not ocr_service.is_available:
        raise PDFExtractionError(
            "OCR is required for this PDF, but the local OCR engine (PaddleOCR) "
            "is unavailable."
        )

    try:
        logger.info(f"Rendering PDF page {page_number} for OCR (DPI={dpi})...")
        t_render = time.perf_counter()
        image_bytes = _render_page_to_png_bytes(page, dpi)
        render_duration = time.perf_counter() - t_render

        try:
            pt_w = getattr(page.rect, "width", 595.0)
            pt_h = getattr(page.rect, "height", 842.0)
            w_px = int(round(pt_w * dpi / 72.0))
            h_px = int(round(pt_h * dpi / 72.0))
        except Exception:
            w_px, h_px = 0, 0

        logger.info(
            f"Page {page_number} rendered image dimensions: {w_px}x{h_px} "
            f"({len(image_bytes)} bytes, {render_duration:.2f}s)."
        )

        t_ocr = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(ocr_service.ocr_image_bytes, image_bytes)
            try:
                raw_text = future.result(timeout=PAGE_OCR_TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError as exc:
                logger.error(
                    f"OCR inference timed out after {PAGE_OCR_TIMEOUT_SECONDS}s on page {page_number}."
                )
                raise OCRProcessingError(
                    f"OCR inference timed out after {PAGE_OCR_TIMEOUT_SECONDS}s on page {page_number}."
                ) from exc

        ocr_duration = time.perf_counter() - t_ocr
        logger.info(f"OCR inference finished for page {page_number} in {ocr_duration:.2f}s.")
    except OCRUnavailableError as exc:
        raise PDFExtractionError(
            "OCR is required for this PDF, but the local OCR engine (PaddleOCR) "
            "is unavailable."
        ) from exc
    except OCRProcessingError as exc:
        logger.error(f"OCR processing failed for page {page_number}: {exc}")
        raise PDFExtractionError("OCR processing failed for this PDF.") from exc
    except Exception as exc:
        logger.error(
            f"Page rendering or OCR failed for page {page_number}: {type(exc).__name__}: {exc}"
        )
        raise PDFExtractionError("A PDF page could not be processed for OCR.") from exc

    cleaned = clean_page_text(raw_text)
    logger.info(f"OCR completed for page {page_number}.")
    return cleaned


def extract_pdf_text(file_path: str, *, use_ocr: bool = False, ocr_dpi: int = 200) -> Dict[str, Any]:
    """Extract text from a PDF page by page, fully locally.

    PyMuPDF is always tried first. When ``use_ocr`` is enabled, only pages
    without meaningful extractable text (scanned/image-only/blank) are rendered
    and transcribed with the local PaddleOCR engine — text pages are never
    sent through OCR.

    Args:
        file_path: Path to the PDF file on the local filesystem.
        use_ocr: When True, scanned/image-only pages fall back to PaddleOCR.
        ocr_dpi: Rendering resolution (DPI) used for pages that need OCR.

    Returns:
        Structured dict with the document filename, page_count and a list of
        per-page dicts (1-based page_number, cleaned text, character_count,
        has_text). Each page also carries an informational ``source`` key:
        "pymupdf" for native text layer extraction, "ocr" for PaddleOCR.
        Pages where neither method found text keep text="" and has_text=False.

    Raises:
        PDFExtractionError: If the file is missing, empty, corrupted,
            password protected, or otherwise unreadable as a PDF; or if OCR is
            required but unavailable/failing.
    """
    started = time.perf_counter()

    if not os.path.isfile(file_path):
        raise PDFExtractionError("PDF file not found on local storage.")

    filename = os.path.basename(file_path)
    logger.info(f"PDF processing started for '{filename}'.")

    # Read locally and open from memory so PyMuPDF never holds an OS file
    # handle on our file. This keeps temporary-file cleanup deterministic,
    # even when the document is corrupted (failed fitz.open on a path can
    # leak the handle on Windows and block deletion).
    try:
        with open(file_path, "rb") as handle:
            pdf_bytes = handle.read()
    except OSError as exc:
        logger.error(f"Failed to read PDF file '{filename}': {type(exc).__name__}")
        raise PDFExtractionError("PDF file could not be read from local storage.") from exc

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        logger.error(f"Failed to open PDF '{filename}': {type(exc).__name__}")
        raise PDFExtractionError("Invalid or corrupted PDF file.") from exc

    try:
        if doc.needs_pass:
            logger.warning(f"PDF '{filename}' is password protected.")
            raise PDFExtractionError("PDF is password protected and cannot be read.")

        pages: List[Dict[str, Any]] = []
        for index, page in enumerate(doc):
            page_number = index + 1  # expose 1-based page numbers
            text = clean_page_text(page.get_text("text"))
            source = SOURCE_PYMUPDF

            # Meaningful native text is always preferred — only pages without
            # any extractable text are sent through the local OCR engine.
            if not text and use_ocr:
                source = SOURCE_OCR
                text = _extract_page_text_via_ocr(page, page_number, ocr_dpi)

            pages.append(
                {
                    "page_number": page_number,
                    "text": text,
                    "character_count": len(text),
                    "has_text": bool(text),
                    "source": source,
                }
            )

        result: Dict[str, Any] = {
            "filename": filename,
            "page_count": len(pages),
            "pages": pages,
        }

        duration_ms = (time.perf_counter() - started) * 1000
        logger.info(
            f"PDF processing completed for '{filename}': {len(pages)} pages "
            f"({duration_ms:.1f} ms)."
        )
        return result
    except PDFExtractionError:
        raise
    except Exception as exc:
        logger.error(f"Failed while reading PDF '{filename}': {type(exc).__name__}")
        raise PDFExtractionError("PDF could not be processed.") from exc
    finally:
        doc.close()