from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image

from ingestion.ocr import OcrError, ocr_image


def _tsv(*words):
    header = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\t"
        "width\theight\tconf\ttext\n"
    )
    rows = [
        f"5\t1\t1\t1\t1\t{index}\t{left}\t{top}\t{width}\t{height}\t96\t{word}\n"
        for index, (word, left, top, width, height) in enumerate(words, start=1)
    ]
    return header + "".join(rows)


def test_ocr_returns_text_page_and_pixel_bounding_box(tmp_path):
    image_path = tmp_path / "scan.png"
    Image.new("RGB", (100, 50), "white").save(image_path)
    output = _tsv(("Annual", 10, 5, 30, 8), ("Report", 45, 5, 35, 8))

    with patch("ingestion.ocr.shutil.which", return_value="tesseract.exe"), patch(
        "ingestion.ocr.subprocess.run",
        return_value=SimpleNamespace(stdout=output),
    ) as run:
        result = ocr_image(str(image_path), page_number=2)

    assert result == {
        "text": "Annual Report",
        "tokens": [
            {"text": "Annual", "bbox": [10, 5, 40, 13]},
            {"text": "Report", "bbox": [45, 5, 80, 13]},
        ],
        "page_number": 2,
        "bbox": [10, 5, 80, 13],
        "page_width": 100,
        "page_height": 50,
        "coordinate_system": {"origin": "top-left", "units": "pixels"},
        "extraction_method": "tesseract_ocr",
    }
    command = run.call_args.args[0]
    assert command[0] == "tesseract.exe"
    assert command[2:] == [
        "stdout",
        "-l",
        "eng",
        "tsv",
    ]


def test_ocr_reports_missing_tesseract_without_running_subprocess(tmp_path):
    image_path = tmp_path / "scan.png"
    Image.new("RGB", (10, 10), "white").save(image_path)

    with patch("ingestion.ocr.shutil.which", return_value=None), patch(
        "ingestion.ocr.subprocess.run"
    ) as run:
        with pytest.raises(OcrError, match="Tesseract was not found on PATH"):
            ocr_image(str(image_path), page_number=1)

    run.assert_not_called()


def test_ocr_returns_none_when_tesseract_finds_no_words(tmp_path):
    image_path = tmp_path / "blank.png"
    Image.new("RGB", (10, 10), "white").save(image_path)
    empty_tsv = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
    with patch("ingestion.ocr.shutil.which", return_value="tesseract.exe"), patch(
        "ingestion.ocr.subprocess.run",
        return_value=SimpleNamespace(stdout=empty_tsv),
    ):
        assert ocr_image(str(image_path), page_number=1) is None


def test_ocr_reports_tesseract_execution_errors(tmp_path):
    image_path = tmp_path / "scan.png"
    Image.new("RGB", (10, 10), "white").save(image_path)
    failure = subprocess.CalledProcessError(
        1, ["tesseract"], stderr="language data unavailable"
    )
    with patch("ingestion.ocr.shutil.which", return_value="tesseract.exe"), patch(
        "ingestion.ocr.subprocess.run", side_effect=failure
    ):
        with pytest.raises(OcrError, match="language data unavailable"):
            ocr_image(str(image_path), page_number=4)
