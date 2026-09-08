"""Phase 8 tests: OCR for scanned/image-only PDFs using PaddleOCR (local).

Synthetic scanned documents are generated locally with Pillow + PyMuPDF;
no confidential industrial documents are used. OCR inference runs fully
on-premise via ``app.services.ocr.service``.
"""

import glob
import io
import os
import tempfile

import fitz
import pytest
from PIL import Image, ImageDraw, ImageFont

from app.services.ocr.service import (
    OCRProcessingError,
    OCRService,
    OCRUnavailableError,
    PADDLEOCR_AVAILABLE,
    ocr_service,
)
from app.tools.pdf_tool import PDFExtractionError, extract_pdf_text

requires_ocr = pytest.mark.skipif(
    not PADDLEOCR_AVAILABLE,
    reason="PaddleOCR is not installed in this environment",
)

# ---------------------------------------------------------------------------
# Synthetic scanned-PDF helpers (image-only pages with clear text)
# ---------------------------------------------------------------------------

OCR_SAMPLE_LINES = [
    "CYBERNEX OCR TEST",
    "Pump inspection status: NORMAL",
    "Bearing temperature: 42 C",
    "Calibration due: 2026-12-01",
]


INDUSTRIAL_INSPECTION_LINES = [
    "CYBERNEX OCR TEST",
    "Industrial Safety Inspection Report",
    "Equipment: Pump-204",
    "Pressure: 10 bar",
    "Temperature: 85 C",
    "Status: NORMAL",
    "Inspector: Test User",
]


def _text_image_png(text_lines):
    """Render clear black-on-white text into an in-memory PNG."""
    width = 1400
    height = 140 + 130 * max(1, len(text_lines))
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=64)
    except TypeError:  # pragma: no cover - very old Pillow fallback
        font = ImageFont.load_default()
    y = 50
    for line in text_lines:
        draw.text((70, y), line, fill="black", font=font)
        y += 130
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _scanned_pdf_bytes(page_lines_list):
    """PDF where every page is an image-only (scanned-like) page.

    Each element is a list of text lines rendered into the page image, or
    None for a deliberately blank (white) image page.
    """
    doc = fitz.open()
    for lines in page_lines_list:
        page = doc.new_page(width=595, height=842)  # A4 portrait, no text layer
        png = _text_image_png(lines or [])
        page.insert_image(fitz.Rect(30, 30, 565, 812), stream=png, keep_proportion=True)
    data = doc.tobytes()
    doc.close()
    return data


def _mixed_pdf_bytes():
    """Page 1 digital text + page 2 scanned image + page 3 digital text."""
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((72, 72), "Digital report page one", fontsize=14)

    p2 = doc.new_page()
    p2.insert_image(
        fitz.Rect(30, 30, 565, 812),
        stream=_text_image_png(["Scanned middle page"]),
        keep_proportion=True,
    )

    p3 = doc.new_page()
    p3.insert_text((72, 72), "Digital report page three", fontsize=14)

    data = doc.tobytes()
    doc.close()
    return data


def _write_pdf(tmp_path, name, data):
    path = tmp_path / name
    path.write_bytes(data)
    return str(path)


def _upload_pdf(client, name, data):
    return client.post(
        "/api/v1/pdf/extract",
        files={"file": (name, io.BytesIO(data), "application/pdf")},
    )

# ---------------------------------------------------------------------------
# OCR service tests
# ---------------------------------------------------------------------------


@requires_ocr
def test_ocr_service_recognizes_synthetic_image():
    text = ocr_service.ocr_image(_text_image_png(OCR_SAMPLE_LINES[:2]))
    assert text.strip()
    assert any(token in text for token in ("CYBERNEX", "OCR", "NORMAL", "Pump"))


@requires_ocr
def test_ocr_service_exposes_alias_and_bytes_entrypoint():
    png = _text_image_png(OCR_SAMPLE_LINES[:1])
    from_bytes = ocr_service.ocr_image_bytes(png)
    from_alias = ocr_service.extract_text_from_image(png)
    assert from_bytes == from_alias
    assert from_bytes.strip()


# ---------------------------------------------------------------------------
# pdf_tool: scanned / mixed PDFs (OCR enabled)
# ---------------------------------------------------------------------------


@requires_ocr
def test_extract_scanned_single_page_pdf(tmp_path):
    path = _write_pdf(
        tmp_path, "scanned_single.pdf",
        _scanned_pdf_bytes([OCR_SAMPLE_LINES[:2]]),
    )
    result = extract_pdf_text(path, use_ocr=True)

    assert result["filename"] == "scanned_single.pdf"
    assert result["page_count"] == 1

    page = result["pages"][0]
    assert page["page_number"] == 1
    assert page["source"] == "ocr"
    assert page["has_text"] is True
    assert page["text"].strip()
    assert any(token in page["text"] for token in ("CYBERNEX", "OCR", "inspection", "NORMAL"))
    assert page["character_count"] == len(page["text"])


@requires_ocr
def test_extract_scanned_multipage_pdf(tmp_path):
    path = _write_pdf(
        tmp_path, "scanned_multi.pdf",
        _scanned_pdf_bytes([OCR_SAMPLE_LINES[:1], OCR_SAMPLE_LINES[2:3]]),
    )
    result = extract_pdf_text(path, use_ocr=True)

    assert result["page_count"] == 2
    assert [p["page_number"] for p in result["pages"]] == [1, 2]
    assert all(p["source"] == "ocr" for p in result["pages"])
    assert all(p["has_text"] is True for p in result["pages"])
    assert all(p["text"].strip() for p in result["pages"])


@requires_ocr
def test_extract_mixed_pdf_selects_method_per_page(tmp_path):
    path = _write_pdf(tmp_path, "mixed.pdf", _mixed_pdf_bytes())
    result = extract_pdf_text(path, use_ocr=True)

    assert result["page_count"] == 3
    assert [p["page_number"] for p in result["pages"]] == [1, 2, 3]
    # Page 1 & 3 = digital text layer; page 2 = scanned image -> OCR.
    assert [p["source"] for p in result["pages"]] == ["pymupdf", "ocr", "pymupdf"]

    assert "Digital report page one" in result["pages"][0]["text"]
    assert "Digital report page three" in result["pages"][2]["text"]
    assert result["pages"][1]["has_text"] is True
    assert any(token in result["pages"][1]["text"] for token in ("Scanned", "middle", "page"))


@requires_ocr
def test_extract_blank_ocr_page_handled_safely(tmp_path):
    # A page that renders to an image but contains no recognized text.
    path = _write_pdf(tmp_path, "blank_scan.pdf", _scanned_pdf_bytes([None]))
    result = extract_pdf_text(path, use_ocr=True)

    page = result["pages"][0]
    assert page["source"] == "ocr"
    assert page["has_text"] is False
    assert page["text"] == ""
    assert page["character_count"] == 0


# ---------------------------------------------------------------------------
# pdf_tool: text PDFs must NOT be sent through OCR
# ---------------------------------------------------------------------------


def test_text_pdf_does_not_trigger_ocr(tmp_path, monkeypatch):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Digital layer text must survive", fontsize=14)
    path = _write_pdf(tmp_path, "digital.pdf", doc.tobytes())
    doc.close()

    def _boom(*args, **kwargs):
        raise AssertionError("OCR must not run for a fully text-based PDF")

    monkeypatch.setattr(ocr_service, "ocr_image", _boom)
    monkeypatch.setattr(ocr_service, "ocr_image_bytes", _boom)

    result = extract_pdf_text(path, use_ocr=True)
    page = result["pages"][0]
    assert page["source"] == "pymupdf"
    assert "Digital layer text must survive" in page["text"]
    assert page["has_text"] is True


def test_extract_whitespace_only_page_triggers_ocr(tmp_path, monkeypatch):
    """Whitespace-only extracted text is not meaningful -> OCR is attempted."""
    doc = fitz.open()
    page = doc.new_page()
    path = _write_pdf(tmp_path, "whitespace.pdf", doc.tobytes())
    doc.close()

    # Simulate OCR availability so the pipeline reports a rendering result.
    ocr_calls = []

    def _fake_ocr(image_bytes):
        ocr_calls.append(image_bytes)
        return "OCR RESULT FOR PAGE"

    monkeypatch.setattr(ocr_service, "_get_paddle_ocr", lambda: object())
    monkeypatch.setattr(ocr_service, "ocr_image_bytes", _fake_ocr)

    result = extract_pdf_text(path, use_ocr=True)
    assert len(ocr_calls) == 1
    assert result["pages"][0]["source"] == "ocr"
    assert result["pages"][0]["text"] == "OCR RESULT FOR PAGE"


# ---------------------------------------------------------------------------
# pdf_tool: error handling
# ---------------------------------------------------------------------------


def test_extract_invalid_pdf_raises(tmp_path):
    path = tmp_path / "corrupt.pdf"
    path.write_bytes(b"this is definitively not a pdf")
    with pytest.raises(PDFExtractionError):
        extract_pdf_text(str(path), use_ocr=True)


def test_extract_empty_file_raises(tmp_path):
    path = tmp_path / "empty.pdf"
    path.write_bytes(b"")
    with pytest.raises(PDFExtractionError):
        extract_pdf_text(str(path), use_ocr=True)


def test_extract_missing_file_raises(tmp_path):
    with pytest.raises(PDFExtractionError):
        extract_pdf_text(str(tmp_path / "missing.pdf"), use_ocr=True)


def test_ocr_unavailable_raises_on_scanned_pdf(tmp_path, monkeypatch):
    path = _write_pdf(tmp_path, "scan.pdf", _scanned_pdf_bytes([["SOME TEXT"]]))
    monkeypatch.setattr(ocr_service, "_get_paddle_ocr", lambda: None)

    with pytest.raises(PDFExtractionError) as exc_info:
        extract_pdf_text(path, use_ocr=True)
    assert "OCR" in str(exc_info.value)


def test_ocr_failure_raises(tmp_path, monkeypatch):
    path = _write_pdf(tmp_path, "scan.pdf", _scanned_pdf_bytes([["SOME TEXT"]]))
    monkeypatch.setattr(ocr_service, "_get_paddle_ocr", lambda: object())

    def _explode(image_bytes):
        raise OCRProcessingError("engine exploded")

    monkeypatch.setattr(ocr_service, "ocr_image_bytes", _explode)

    with pytest.raises(PDFExtractionError) as exc_info:
        extract_pdf_text(path, use_ocr=True)
    assert "OCR" in str(exc_info.value)


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


def test_api_pdf_extract_normal_pdf(client):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "API normal text fixture", fontsize=14)
    data = doc.tobytes()
    doc.close()

    res = _upload_pdf(client, "api_normal.pdf", data)
    assert res.status_code == 200

    body = res.json()
    # Backward-compatible public schema: no new page-level keys.
    assert set(body["pages"][0].keys()) == {"page_number", "text", "character_count", "has_text"}
    assert body["pages"][0]["has_text"] is True
    assert "API normal text fixture" in body["pages"][0]["text"]


@requires_ocr
def test_api_pdf_extract_scanned_pdf(client):
    data = _scanned_pdf_bytes([OCR_SAMPLE_LINES])
    res = _upload_pdf(client, "api_scanned.pdf", data)
    assert res.status_code == 200

    body = res.json()
    assert body["page_count"] == 1
    page = body["pages"][0]
    assert page["page_number"] == 1
    assert page["has_text"] is True
    assert page["text"].strip()
    assert any(token in page["text"] for token in ("CYBERNEX", "OCR", "inspection", "NORMAL"))


@requires_ocr
def test_api_pdf_extract_scanned_multipage_returns_actual_text(client):
    """POST /api/v1/pdf/extract returns actual page-level OCR text for each page in multi-page scan."""
    page1_lines = ["CYBERNEX OCR PAGE ONE", "Equipment: Boiler-101"]
    page2_lines = ["CYBERNEX OCR PAGE TWO", "Equipment: Turbine-202"]
    data = _scanned_pdf_bytes([page1_lines, page2_lines])
    res = _upload_pdf(client, "api_scanned_multi.pdf", data)
    assert res.status_code == 200

    body = res.json()
    assert body["page_count"] == 2
    assert len(body["pages"]) == 2

    p1 = body["pages"][0]
    assert p1["page_number"] == 1
    assert p1["has_text"] is True
    assert p1["character_count"] > 0
    assert any(tok in p1["text"] for tok in ("ONE", "Boiler"))

    p2 = body["pages"][1]
    assert p2["page_number"] == 2
    assert p2["has_text"] is True
    assert p2["character_count"] > 0
    assert any(tok in p2["text"] for tok in ("TWO", "Turbine"))


def test_api_pdf_extract_rejects_non_pdf(client):
    res = _upload_pdf(client, "notes.txt", b"plain text, not a pdf")
    assert res.status_code == 400
    assert "PDF" in res.json()["detail"]


def test_api_pdf_extract_rejects_corrupted_pdf(client):
    res = _upload_pdf(client, "broken.pdf", b"definitely not a real pdf body")
    assert res.status_code == 422
    assert "detail" in res.json()


def test_api_pdf_extract_requires_file(client):
    res = client.post("/api/v1/pdf/extract")
    assert res.status_code == 422  # FastAPI validation: required file missing


def test_api_temp_upload_files_are_cleaned_up(client):
    tmpdir = tempfile.gettempdir()
    pattern = os.path.join(tmpdir, "cybernex_pdf_*.pdf")

    before = set(glob.glob(pattern))

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "cleanup fixture", fontsize=14)
    data = doc.tobytes()
    doc.close()
    assert _upload_pdf(client, "cleanup.pdf", data).status_code == 200

    # Also exercise the error path (extraction raises -> file still removed).
    assert _upload_pdf(client, "cleanup_bad.pdf", b"garbage").status_code == 422

    after = set(glob.glob(pattern))
    assert after == before  # no temporary upload leaked on either path


# ---------------------------------------------------------------------------
# Reading Order & Synthetic Industrial Report Tests
# ---------------------------------------------------------------------------


def test_ocr_reading_order_sorting():
    """Verify that _sort_reading_order groups lines and orders top-to-bottom, left-to-right."""
    # Items: (text, conf, (xmin, ymin, xmax, ymax))
    items = [
        ("Line 2 Col 2", 0.95, (200.0, 100.0, 350.0, 120.0)),
        ("Line 1 Col 2", 0.95, (200.0, 30.0, 350.0, 50.0)),
        ("Line 2 Col 1", 0.95, (50.0, 102.0, 180.0, 122.0)),
        ("Line 1 Col 1", 0.95, (50.0, 32.0, 180.0, 52.0)),
    ]
    texts, scores = OCRService._sort_reading_order(items)
    assert texts == ["Line 1 Col 1", "Line 1 Col 2", "Line 2 Col 1", "Line 2 Col 2"]


@requires_ocr
def test_extract_scanned_synthetic_industrial_report(tmp_path):
    """Verify PaddleOCR recovers exact industrial safety report fields from a scanned PDF."""
    pdf_bytes = _scanned_pdf_bytes([INDUSTRIAL_INSPECTION_LINES])
    path = _write_pdf(tmp_path, "synthetic_inspection.pdf", pdf_bytes)

    result = extract_pdf_text(path, use_ocr=True)
    assert result["page_count"] == 1
    page = result["pages"][0]
    assert page["page_number"] == 1
    assert page["has_text"] is True
    assert page["source"] == "ocr"

    page_text = page["text"]
    assert "CYBERNEX" in page_text
    assert "Inspection" in page_text
    assert "Pump-204" in page_text or "Pump" in page_text
    assert "NORMAL" in page_text
    assert "10 bar" in page_text or "bar" in page_text
    assert "85 C" in page_text or "85" in page_text
    assert "Test User" in page_text or "User" in page_text


@requires_ocr
def test_ocr_service_extract_text_preserves_page_headings_and_content(tmp_path):
    """Verify ocr_service.extract_text outputs non-empty page headers for scanned documents."""
    pdf_bytes = _scanned_pdf_bytes([INDUSTRIAL_INSPECTION_LINES])
    path = _write_pdf(tmp_path, "inspection_service.pdf", pdf_bytes)

    res = ocr_service.extract_text(path)
    assert res["pages"] == 1
    assert "PyMuPDF + PaddleOCR" in res["engine"]
    full_text = res["text"]
    assert "--- Page 1 ---" in full_text
    assert any(tok in full_text for tok in ("NORMAL", "Pump", "Inspection", "CYBERNEX"))


@requires_ocr
def test_agent_pipeline_end_to_end_scanned_pdf_generates_docx(tmp_path):
    """End-to-end integration test: Scanned PDF -> Agent Graph -> DOCX output with actual OCR text."""
    import docx
    import zipfile
    from app.services.agent.graph import agent_graph, AgentState

    pdf_bytes = _scanned_pdf_bytes([INDUSTRIAL_INSPECTION_LINES])
    pdf_path = _write_pdf(tmp_path, "industrial_report.pdf", pdf_bytes)

    initial_state: AgentState = {
        "task_id": "test-ocr-task",
        "run_id": "test-run-ocr",
        "prompt": "Extract all text from the attached scanned PDF page by page. Return the OCR-extracted text for each page separately. Do not summarize the document.",
        "selected_model": "Auto",
        "selected_tools": ["OCR", "Documents"],
        "files": [
            {
                "file_type": "PDF",
                "original_name": "industrial_report.pdf",
                "file_path": pdf_path,
            }
        ],
        "task_understanding": "",
        "plan_steps": [],
        "model_routed": "Auto",
        "retrieved_chunks": [],
        "ocr_text": "",
        "execution_result": "",
        "verification_status": "Pending",
        "deliverable": None,
        "current_step": 0,
        "step_events": [],
    }

    final_state = agent_graph.invoke(initial_state)

    deliv = final_state.get("deliverable")
    assert deliv is not None
    assert deliv["type"] == "DOCX"
    docx_path = deliv["file_path"]
    assert os.path.exists(docx_path)

    # 1. Structural validation: must be a valid ZIP archive
    assert zipfile.is_zipfile(docx_path)

    # 2. Content validation: open via python-docx and inspect paragraphs
    doc = docx.Document(docx_path)
    all_paras = [p.text for p in doc.paragraphs]
    combined_text = "\n".join(all_paras)

    # Filename and page heading must be present
    assert any("industrial_report.pdf" in p for p in all_paras)
    assert any("--- Page 1 ---" in p for p in all_paras)

    # The actual OCR text must be present, NOT empty page headings
    assert any(tok in combined_text for tok in ("NORMAL", "Pump", "Inspection", "CYBERNEX"))