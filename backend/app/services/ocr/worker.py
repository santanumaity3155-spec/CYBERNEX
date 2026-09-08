"""Subprocess worker for running PaddleOCR locally in the backend virtual environment.

This allows CYBERNEX to transparently execute local PaddleOCR inference even if
the FastAPI server was launched under a Python interpreter (such as Python 3.14)
that lacks native paddleocr wheel support.
"""

import io
import json
import sys
import os

# Ensure backend root is on sys.path so app imports succeed
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


def main():
    try:
        from app.services.ocr.service import ocr_service

        raw = sys.stdin.buffer.read()
        if not raw:
            payload = json.dumps({"text": "", "score": None, "success": True})
            sys.stdout.write(f"\n__CYBERNEX_JSON_START__{payload}__CYBERNEX_JSON_END__\n")
            sys.stdout.flush()
            return

        text, score = ocr_service._text_and_score(raw)
        payload = json.dumps({"text": text, "score": score, "success": True})
        sys.stdout.write(f"\n__CYBERNEX_JSON_START__{payload}__CYBERNEX_JSON_END__\n")
        sys.stdout.flush()
    except Exception as exc:
        payload = json.dumps({"error": str(exc), "success": False})
        sys.stdout.write(f"\n__CYBERNEX_JSON_START__{payload}__CYBERNEX_JSON_END__\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
