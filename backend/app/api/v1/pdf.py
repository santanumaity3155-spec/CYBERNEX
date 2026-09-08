"""PDF processing endpoints (Phase 7 + Phase 8 OCR integration).

Accepts a PDF upload, writes it to a temporary local file, extracts page-level
text and returns structured JSON.

- Normal text-based pages are extracted with PyMuPDF.
- Scanned/image-only pages (no meaningful extractable text) are rendered
  in-memory and transcribed with the local PaddleOCR engine, page by page.
- OCR runs fully on-premise; uploaded files are temporary and always
  cleaned up, and nothing is sent outside the machine.
"""

import asyncio
import os
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.config import get_settings
from app.core.logging import logger
from app.schemas.pdf import PDFExtractionResponse
from app.tools.pdf_tool import PDFExtractionError, extract_pdf_text

settings = get_settings()
router = APIRouter(prefix="/pdf", tags=["PDF"])

UPLOAD_CHUNK_SIZE_BYTES = 1024 * 1024  # stream uploads in 1 MB chunks


@router.post(
    "/extract",
    response_model=PDFExtractionResponse,
    summary="Extract Page-Level Text from PDF",
)
async def extract_pdf(file: UploadFile = File(...)):
    """Extract page-level text from an uploaded PDF.

    Text-based pages are handled by PyMuPDF; scanned/image-only pages are
    processed automatically with the local PaddleOCR engine.
    """
    filename = file.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext != "pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type. Only PDF files are accepted.",
        )

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    logger.info(f"PDF extraction requested for '{filename}'.")

    tmp_fd, tmp_path = tempfile.mkstemp(prefix="cybernex_pdf_", suffix=".pdf")
    try:
        with os.fdopen(tmp_fd, "wb") as buffer:
            size_bytes = 0
            while chunk := await file.read(UPLOAD_CHUNK_SIZE_BYTES):
                size_bytes += len(chunk)
                if size_bytes > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=(
                            "PDF exceeds the maximum upload size of "
                            f"{settings.MAX_UPLOAD_SIZE_MB} MB."
                        ),
                    )
                buffer.write(chunk)

        try:
            # Phase 8: scanned/image-only pages automatically fall back to the
            # local PaddleOCR engine; text pages still use PyMuPDF only.
            # Run in worker thread so CPU OCR inference never freezes the event loop.
            result = await asyncio.to_thread(extract_pdf_text, tmp_path, use_ocr=True)
        except PDFExtractionError as exc:
            logger.warning(f"PDF extraction failed for '{filename}': {exc}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            )
    finally:
        await file.close()
        try:
            os.remove(tmp_path)
        except OSError:
            pass  # temporary file cleanup is best-effort

    logger.info(
        f"PDF extraction endpoint completed for '{filename}' "
        f"({result['page_count']} pages, {size_bytes} bytes)."
    )

    return PDFExtractionResponse(
        filename=filename,
        page_count=result["page_count"],
        pages=result["pages"],
    )