"""Optional OCR for pages without useful text; Tesseract is an external tool."""

from __future__ import annotations

import csv
import io
import shutil
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image


class OcrError(RuntimeError):
    """Raised when requested OCR cannot be completed."""


def ocr_image(image_path: str, page_number: int, language: str = "eng") -> dict[str, Any] | None:
    """OCR an image with Tesseract and return text plus a page-pixel bounding box."""
    tesseract = shutil.which("tesseract")
    if not tesseract:
        raise OcrError(
            "OCR was requested, but Tesseract was not found on PATH. Install Tesseract "
            "and reopen PowerShell, or omit --ocr."
        )

    image_file = Path(image_path)
    if not image_file.is_file():
        raise OcrError(f"OCR source image does not exist: {image_file}")
    try:
        with Image.open(image_file) as image:
            image_width, image_height = image.size
        result = subprocess.run(
            [tesseract, str(image_file), "stdout", "-l", language, "tsv"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip()
        raise OcrError(f"Tesseract failed on page {page_number}: {detail}") from exc
    except OSError as exc:
        raise OcrError(f"Could not read OCR image for page {page_number}: {exc}") from exc

    words: list[str] = []
    boxes: list[tuple[int, int, int, int]] = []
    try:
        for row in csv.DictReader(io.StringIO(result.stdout), delimiter="\t"):
            text = (row.get("text") or "").strip()
            if not text:
                continue
            left = int(row["left"])
            top = int(row["top"])
            width = int(row["width"])
            height = int(row["height"])
            if width <= 0 or height <= 0:
                continue
            words.append(text)
            boxes.append((left, top, left + width, top + height))
    except (KeyError, TypeError, ValueError) as exc:
        raise OcrError(f"Tesseract returned invalid box data for page {page_number}.") from exc

    text = " ".join(words).strip()
    if not text:
        return None

    bbox = [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]
    return {
        "text": text,
        "page_number": page_number,
        "bbox": bbox,
        "page_width": image_width,
        "page_height": image_height,
        "coordinate_system": {"origin": "top-left", "units": "pixels"},
        "extraction_method": "tesseract_ocr",
    }
