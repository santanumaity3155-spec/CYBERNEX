"""
Local OCR service (Phase 8).

PaddleOCR-based, fully on-premise text extraction from images and PDFs.

Sovereignty guarantees:
- OCR inference runs 100% locally. No cloud OCR, no external APIs, no
  upload of images/documents to the internet.
- The PaddleOCR inference engine is initialized once per process and reused
  across pages (lazy loading, cached on the service instance).

Behavior:
- text-based PDF pages are read with PyMuPDF and are NOT sent to OCR;
- pages without meaningful extractable text are rendered (in-memory, PNG)
  and processed with PaddleOCR;
- rendered PNG bytes are decoded with OpenCV in memory — no temporary image
  files are created or left behind.

Supported engine versions:
- PaddleOCR 3.x (``PaddleOCR(..., enable_mkldnn=False)``) — verified working
  on Windows with PaddlePaddle 3.3.x (CPU).
- PaddleOCR 2.x (``use_angle_cls`` API) — supported as a fallback path.
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import fitz  # PyMuPDF

from app.core.logging import logger

try:
    from paddleocr import PaddleOCR
    import paddleocr as _paddleocr_pkg

    PADDLEOCR_AVAILABLE = True
    _PADDLEOCR_MAJOR = int(str(getattr(_paddleocr_pkg, "__version__", "0")).split(".")[0])
except Exception:  # pragma: no cover - depends on environment
    PADDLEOCR_AVAILABLE = False
    _PADDLEOCR_MAJOR = 0

# Reasonable CPU-friendly resolution for OCR page rendering (dots per inch).
OCR_RENDER_DPI = 200


def _find_venv_python() -> Optional[str]:
    """Find a Python interpreter in the local backend virtual environment."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        candidate_win = os.path.join(cur_dir, ".venv", "Scripts", "python.exe")
        candidate_posix = os.path.join(cur_dir, ".venv", "bin", "python")
        if os.path.isfile(candidate_win):
            return candidate_win
        if os.path.isfile(candidate_posix):
            return candidate_posix
        parent = os.path.dirname(cur_dir)
        if parent == cur_dir:
            break
        cur_dir = parent
    return None


class OCRUnavailableError(RuntimeError):
    """Raised when the local OCR engine is not installed or cannot be initialized."""


class OCRProcessingError(RuntimeError):
    """Raised when OCR inference fails on a provided image.

    The message is intentionally generic and safe to expose; technical
    details are only written to the local log.
    """


class OCRService:
    """Reusable local OCR facade backed by PaddleOCR."""

    def __init__(self) -> None:
        self._ocr_engine: Any = None
        self._paddle_api_major: int = _PADDLEOCR_MAJOR

    # ------------------------------------------------------------------
    # Availability / engine lifecycle
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """True when a local PaddleOCR engine can be (and was) initialized."""
        if self._get_paddle_ocr() is not None:
            return True
        # If in-process is unavailable (e.g. running under Python 3.14),
        # check if .venv Python has PaddleOCR available for subprocess execution.
        if not PADDLEOCR_AVAILABLE:
            venv_py = _find_venv_python()
            return venv_py is not None and os.path.isfile(venv_py)
        return False

    def _engine_kwargs(self) -> Dict[str, Any]:
        """Kwargs used to construct PaddleOCR, adapted to the installed API.

        The 3.x pipeline needs MKLDNN disabled on Windows/PaddlePaddle 3.3.x:
        the oneDNN backend hits an upstream 'ConvertPirAttribute2RuntimeAttribute'
        crash while running the PP-OCRv6 inference programs, so we force the
        plain 'paddle' run mode (still fully local, just CPU-only).

        We also cap text detection max side length to 960 to ensure predictable,
        high-speed CPU inference without compromising document text recognition.

        For CPU-bound inference, PP-OCRv4 mobile models are used instead of the
        heavier PP-OCRv6 medium defaults: they deliver comparable recognition
        accuracy on industrial documents while running ~2-3x faster on CPU.
        """
        if _PADDLEOCR_MAJOR >= 3:
            return {
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": False,
                "lang": "en",
                "ocr_version": "PP-OCRv4",
                "enable_mkldnn": False,
                "text_det_limit_side_len": 960,
                "text_det_limit_type": "max",
                "text_det_box_thresh": 0.7,
            }
        # PaddleOCR 2.x API
        return {"use_angle_cls": True, "lang": "en", "show_log": False}

    def _new_engine(self) -> Any:
        kwargs = self._engine_kwargs()
        try:
            return PaddleOCR(**kwargs)
        except (TypeError, ValueError) as exc:
            # Older 3.x releases may not know 'enable_mkldnn' or tuning kwargs; retry.
            if _PADDLEOCR_MAJOR >= 3:
                logger.warning(
                    "PaddleOCR init with tuned kwargs failed (%s); retrying minimal config.", exc
                )
                try:
                    return PaddleOCR(
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False,
                        lang="en",
                        ocr_version="PP-OCRv4",
                        text_det_limit_side_len=960,
                        text_det_limit_type="max",
                        text_det_box_thresh=0.7,
                    )
                except (TypeError, ValueError):
                    return PaddleOCR(
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False,
                        lang="en",
                        ocr_version="PP-OCRv4",
                    )
            raise

    def _get_paddle_ocr(self) -> Any:
        """Lazily build and cache the PaddleOCR engine (once per process)."""
        if self._ocr_engine is not None:
            return self._ocr_engine
        if not PADDLEOCR_AVAILABLE:
            logger.warning("PaddleOCR is not installed in current process; will check .venv worker.")
            return None
        try:
            self._ocr_engine = self._new_engine()
            logger.info("PaddleOCR engine initialized.")
        except Exception as exc:
            logger.error(f"Failed to initialize PaddleOCR engine: {exc}")
            self._ocr_engine = None
        return self._ocr_engine

    # ------------------------------------------------------------------
    # Result parsing & reading order sorting
    # ------------------------------------------------------------------

    @staticmethod
    def _sort_reading_order(
        items: List[Tuple[str, float, Optional[Tuple[float, float, float, float]]]]
    ) -> Tuple[List[str], List[float]]:
        """Sort detected text lines in natural reading order (top-to-bottom, left-to-right).

        Grouping lines vertically based on median line height prevents
        slight detection slant or column ordering from shuffling the text.
        """
        if not items:
            return [], []

        # If no bounding boxes are available, retain original detection order
        if not any(item[2] is not None for item in items):
            return [it[0] for it in items], [it[1] for it in items]

        # Calculate median line height to cluster lines into vertical reading bands
        heights = [
            (box[3] - box[1])
            for _, _, box in items
            if box is not None and (box[3] - box[1]) > 0
        ]
        line_height = float(np.median(heights)) if heights else 20.0
        line_tolerance = max(line_height * 0.5, 8.0)

        # Primary sort: vertical line bucket (ymin / line_tolerance)
        # Secondary sort: horizontal position (xmin)
        def line_key(
            item: Tuple[str, float, Optional[Tuple[float, float, float, float]]]
        ) -> Tuple[int, float]:
            box = item[2]
            if box is None:
                return (0, 0.0)
            xmin, ymin, _, _ = box
            line_bucket = int(round(ymin / line_tolerance))
            return (line_bucket, xmin)

        sorted_items = sorted(items, key=line_key)
        return [it[0] for it in sorted_items], [it[1] for it in sorted_items]

    def _extract_v3(self, results: Any) -> Tuple[List[str], List[float]]:
        """Parse PaddleOCR 3.x ``predict`` output (dict-like OCRResult objects)."""
        raw_items: List[Tuple[str, float, Optional[Tuple[float, float, float, float]]]] = []
        for res in results:
            if res is None:
                continue

            texts: List[str] = []
            scores: List[float] = []
            boxes_data = None

            if hasattr(res, "get"):
                texts = res.get("rec_texts") or []
                scores = res.get("rec_scores") or []
                boxes_data = res.get("rec_boxes")
                if boxes_data is None:
                    boxes_data = res.get("dt_polys") or res.get("rec_polys")
                if not texts and "text" in res:
                    texts = res["text"]
            elif hasattr(res, "rec_texts"):
                texts = getattr(res, "rec_texts", [])
                scores = getattr(res, "rec_scores", [])
                boxes_data = getattr(res, "rec_boxes", None)

            if isinstance(texts, str):
                texts = [texts]
            if isinstance(scores, (int, float)):
                scores = [scores]

            for idx, item in enumerate(texts or []):
                if not isinstance(item, str) or not item.strip():
                    continue
                conf = 0.95
                if isinstance(scores, (list, tuple)) and idx < len(scores):
                    try:
                        conf = float(scores[idx])
                    except (TypeError, ValueError):
                        pass

                box_tuple = None
                if boxes_data is not None and idx < len(boxes_data):
                    try:
                        b = boxes_data[idx]
                        if hasattr(b, "tolist"):
                            b = b.tolist()
                        if len(b) == 4 and isinstance(b[0], (int, float)):
                            box_tuple = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
                        elif len(b) == 4 and isinstance(b[0], (list, tuple)):
                            xs = [pt[0] for pt in b]
                            ys = [pt[1] for pt in b]
                            box_tuple = (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))
                    except Exception:
                        box_tuple = None

                raw_items.append((item.strip(), conf, box_tuple))

        return self._sort_reading_order(raw_items)

    def _extract_v2(self, result: Any) -> Tuple[List[str], List[float]]:
        """Parse PaddleOCR 2.x ``ocr`` output: [[[box, (text, conf)], ...], ...]."""
        raw_items: List[Tuple[str, float, Optional[Tuple[float, float, float, float]]]] = []
        if not result:
            return [], []
        first = result[0] if isinstance(result, list) else result
        if not first:
            return [], []
        for item in first:
            try:
                poly = item[0]
                text, conf = item[1]
            except (IndexError, TypeError, ValueError):
                continue
            if not isinstance(text, str) or not text.strip():
                continue
            box_tuple = None
            try:
                if poly and len(poly) == 4:
                    xs = [pt[0] for pt in poly]
                    ys = [pt[1] for pt in poly]
                    box_tuple = (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))
            except Exception:
                pass
            try:
                score = float(conf)
            except (TypeError, ValueError):
                score = 0.95
            raw_items.append((text.strip(), score, box_tuple))
        return self._sort_reading_order(raw_items)

    # ------------------------------------------------------------------
    # Core recognition
    # ------------------------------------------------------------------

    def _decode_image(self, image: Any) -> np.ndarray:
        """Convert an image (PNG bytes or ndarray) into a BGR ndarray in memory."""
        if isinstance(image, np.ndarray):
            arr = image
            if arr.ndim == 2:  # grayscale -> BGR
                return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            if arr.ndim == 3 and arr.shape[2] == 4:  # RGBA -> BGR
                return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            if arr.ndim == 3 and arr.shape[2] == 3:
                # PaddleOCR historically loads images with cv2.imread (BGR).
                return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            raise OCRProcessingError("Unsupported image array shape for OCR.")
        if isinstance(image, (bytes, bytearray, memoryview)):
            raw = np.frombuffer(bytes(image), dtype=np.uint8)
            arr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if arr is None:
                raise OCRProcessingError("Could not decode the provided image bytes.")
            return arr
        if isinstance(image, str):  # convenience for local image file paths
            if not os.path.isfile(image):
                raise OCRProcessingError("Image file not found on local storage.")
            arr = cv2.imread(image, cv2.IMREAD_COLOR)
            if arr is None:
                raise OCRProcessingError("Could not read the provided image file.")
            return arr
        raise OCRProcessingError("Unsupported image input type for OCR.")

    def _run_via_venv(self, image: Any) -> Tuple[str, Optional[float]]:
        """Run OCR via worker process in backend/.venv when current python lacks PaddleOCR."""
        venv_py = _find_venv_python()
        if not venv_py or not os.path.isfile(venv_py):
            raise OCRUnavailableError("Local OCR engine (PaddleOCR) is unavailable.")

        if isinstance(image, (bytes, bytearray, memoryview)):
            image_bytes = bytes(image)
        elif isinstance(image, np.ndarray):
            success, encoded = cv2.imencode(".png", image)
            if not success:
                raise OCRProcessingError("Failed to encode image array for OCR.")
            image_bytes = encoded.tobytes()
        elif isinstance(image, str) and os.path.isfile(image):
            with open(image, "rb") as f:
                image_bytes = f.read()
        else:
            raise OCRProcessingError("Unsupported image input format for OCR.")

        worker_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker.py")
        backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        import subprocess
        import json

        try:
            proc = subprocess.run(
                [venv_py, worker_script],
                input=image_bytes,
                capture_output=True,
                cwd=backend_dir,
                timeout=120,
            )
            if proc.returncode != 0:
                logger.error(f"OCR worker subprocess failed: {proc.stderr.decode('utf-8', errors='ignore')}")
                raise OCRProcessingError("OCR inference failed in worker process.")

            stdout_str = proc.stdout.decode("utf-8", errors="ignore")
            if "__CYBERNEX_JSON_START__" in stdout_str and "__CYBERNEX_JSON_END__" in stdout_str:
                json_chunk = stdout_str.split("__CYBERNEX_JSON_START__", 1)[1].split("__CYBERNEX_JSON_END__", 1)[0].strip()
                out_data = json.loads(json_chunk)
            else:
                out_data = json.loads(stdout_str.strip())

            if not out_data.get("success", False):
                err = out_data.get("error", "Unknown error")
                logger.error(f"OCR worker returned error: {err}")
                raise OCRProcessingError(f"OCR inference failed: {err}")

            return out_data.get("text", ""), out_data.get("score")
        except subprocess.TimeoutExpired as exc:
            logger.error("OCR worker timed out.")
            raise OCRProcessingError("OCR inference timed out.") from exc
        except (OCRUnavailableError, OCRProcessingError):
            raise
        except Exception as exc:
            logger.error(f"OCR worker execution error: {type(exc).__name__}: {exc}")
            raise OCRProcessingError("OCR inference worker error.") from exc

    def _text_and_score(self, image: Any) -> Tuple[str, Optional[float]]:
        """Run OCR on an in-memory image; returns (joined_text, mean_confidence)."""
        engine = self._get_paddle_ocr()
        if engine is not None:
            arr = self._decode_image(image)
            try:
                if self._paddle_api_major >= 3:
                    logger.info(
                        f"Starting PaddleOCR predict on image array shape={arr.shape}, dtype={arr.dtype}..."
                    )
                    t_start = time.perf_counter()
                    results = engine.predict(arr)
                    duration = time.perf_counter() - t_start
                    num_results = len(results) if isinstance(results, (list, tuple)) else 1
                    res_type = type(results).__name__
                    logger.info(
                        f"PaddleOCR predict completed in {duration:.2f}s: "
                        f"returned {num_results} result item(s) of type {res_type}."
                    )
                    t_start = time.perf_counter()
                    lines, scores = self._extract_v3(results)
                    parse_duration = time.perf_counter() - t_start
                    logger.info(
                        f"Extracted {len(lines)} OCR text line(s) in reading order in {parse_duration:.2f}s."
                    )
                else:
                    logger.info(
                        f"Starting PaddleOCR v2 ocr on image array shape={arr.shape}, dtype={arr.dtype}..."
                    )
                    t_start = time.perf_counter()
                    result = engine.ocr(arr, cls=True)
                    duration = time.perf_counter() - t_start
                    logger.info(f"PaddleOCR v2 ocr completed in {duration:.2f}s.")
                    t_start = time.perf_counter()
                    lines, scores = self._extract_v2(result)
                    parse_duration = time.perf_counter() - t_start
                    logger.info(
                        f"Extracted {len(lines)} OCR text line(s) in reading order in {parse_duration:.2f}s."
                    )
            except OCRUnavailableError:
                raise
            except Exception as exc:
                logger.error(f"PaddleOCR inference failed: {type(exc).__name__}: {exc}")
                raise OCRProcessingError("OCR inference failed on the provided image.") from exc

            text = "\n".join(lines)
            mean_score = float(np.mean(scores)) if scores else None
            return text, mean_score

        # Fallback to .venv worker if available
        if not PADDLEOCR_AVAILABLE and self.is_available:
            return self._run_via_venv(image)

        raise OCRUnavailableError("Local OCR engine (PaddleOCR) is unavailable.")

    def ocr_image(self, image: Any) -> str:
        """Extract text from an image (PNG bytes or ndarray), fully local.

        This is the primary reusable entry point of the OCR service.
        """
        text, _ = self._text_and_score(image)
        return text

    def extract_text_from_image(self, image: Any) -> str:
        """Alias of :meth:`ocr_image` for clarity in calling code."""
        return self.ocr_image(image)

    def ocr_image_bytes(self, image_bytes: bytes) -> str:
        """Extract text from encoded image bytes (PNG/JPEG), fully local."""
        return self.ocr_image(image_bytes)

    # ------------------------------------------------------------------
    # File-level pipeline (kept for RAG ingestion + /api/v1/ocr)
    # ------------------------------------------------------------------

    def extract_text(self, file_path: str, min_density_chars: int = 50) -> Dict[str, Any]:
        """High-level local text extraction for a stored file.

        PDF   -> PyMuPDF pages first; pages with insufficient text are rendered
                 and run through PaddleOCR.
        IMAGE -> PaddleOCR directly.
        TEXT  -> read as-is.

        Output contract (unchanged from Phase 7 scaffolding):
        {"text": str, "pages": int, "confidence": float, "engine": str}
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found for OCR: {file_path}")

        ext = file_path.rsplit(".", 1)[-1].lower() if "." in os.path.basename(file_path) else ""

        if ext == "pdf":
            return self._extract_pdf(file_path, min_density_chars)

        if ext in ("png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"):
            try:
                with open(file_path, "rb") as handle:
                    image_bytes = handle.read()
                text, mean_score = self._text_and_score(image_bytes)
                return {
                    "text": text,
                    "pages": 1,
                    "confidence": round(mean_score, 4) if mean_score is not None else 0.0,
                    "engine": "PaddleOCR",
                }
            except OCRUnavailableError:
                return {"text": "", "pages": 1, "confidence": 0.0, "engine": "OCR_UNAVAILABLE"}
            except Exception as exc:
                logger.error(f"Image OCR failed for '{file_path}': {type(exc).__name__}")
                return {"text": "", "pages": 1, "confidence": 0.0, "engine": "Error"}

        # Fallback: plain text file reading.
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
                content = handle.read()
            return {"text": content, "pages": 1, "confidence": 1.0, "engine": "TextReader"}
        except Exception as exc:
            logger.error(f"Text read failed for '{file_path}': {type(exc).__name__}")
            return {"text": "", "pages": 0, "confidence": 0.0, "engine": "Error"}

    def _extract_pdf(self, file_path: str, min_density_chars: int = 50) -> Dict[str, Any]:
        """PyMuPDF-first, page-aware OCR fallback for PDF files."""
        from app.tools.pdf_tool import extract_pdf_text, PDFExtractionError

        filename = os.path.basename(file_path)
        try:
            result = extract_pdf_text(file_path, use_ocr=True)
        except PDFExtractionError as exc:
            logger.error(f"Failed to extract PDF '{filename}' for OCR: {exc}")
            raise
        except Exception as exc:
            logger.error(f"Failed to process PDF '{filename}' for OCR: {exc}")
            return {"text": "", "pages": 0, "confidence": 0.0, "engine": "Error"}

        parts: List[str] = []
        ocr_used = False
        for p in result.get("pages", []):
            p_num = p.get("page_number")
            p_text = p.get("text", "")
            if p.get("source") == "ocr" and p.get("has_text"):
                ocr_used = True
            if p_text.strip():
                parts.append(f"--- Page {p_num} ---\n{p_text}")
            else:
                parts.append(f"--- Page {p_num} ---\n")

        full_text = "\n\n".join(parts).strip()
        engine = "PyMuPDF + PaddleOCR" if ocr_used else "PyMuPDF"
        return {
            "text": full_text,
            "pages": result.get("page_count", 0),
            "confidence": 0.95 if full_text else 0.0,
            "engine": engine,
        }


# Process-wide singleton so the PaddleOCR engine (model weights) is loaded once
# and reused across requests/pages — never re-initialized per page.
ocr_service = OCRService()