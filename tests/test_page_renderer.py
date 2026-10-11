import zlib
from pathlib import Path

import pytest
from PIL import Image
from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
    NumberObject,
)

from ingestion.assets import document_asset_key, resolve_asset_path
from ingestion.page_renderer import PageRenderError, render_pdf_pages


def _write_synthetic_pdf(
    path: Path, page_count: int = 2, include_image: bool = False
) -> None:
    writer = PdfWriter()
    for page_number in range(1, page_count + 1):
        page = writer.add_blank_page(width=300, height=400)
        if include_image and page_number == page_count:
            image_stream = DecodedStreamObject()
            image_stream.set_data(zlib.compress(bytes([255, 0, 0] * 4)))
            image_stream[NameObject("/Type")] = NameObject("/XObject")
            image_stream[NameObject("/Subtype")] = NameObject("/Image")
            image_stream[NameObject("/Width")] = NumberObject(2)
            image_stream[NameObject("/Height")] = NumberObject(2)
            image_stream[NameObject("/ColorSpace")] = NameObject("/DeviceRGB")
            image_stream[NameObject("/BitsPerComponent")] = NumberObject(8)
            image_stream[NameObject("/Filter")] = NameObject("/FlateDecode")
            image_reference = writer._add_object(image_stream)
            page[NameObject("/Resources")] = DictionaryObject(
                {
                    NameObject("/XObject"): DictionaryObject(
                        {NameObject("/Im0"): image_reference}
                    )
                }
            )
            stream = DecodedStreamObject()
            stream.set_data(b"q 200 0 0 100 50 150 cm /Im0 Do Q")
            page[NameObject("/Contents")] = writer._add_object(stream)
            continue
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        font_reference = writer._add_object(font)
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {NameObject("/F1"): font_reference}
                )
            }
        )
        stream = DecodedStreamObject()
        stream.set_data(
            b"BT /F1 20 Tf 40 350 Td (Synthetic heading) Tj ET "
            b"0.2 0.5 0.8 rg 40 100 100 80 re f"
        )
        page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as pdf_file:
        writer.write(pdf_file)


def test_render_synthetic_vector_pdf_to_document_scoped_pages(tmp_path, monkeypatch):
    pdf_path = tmp_path / "vector-report.pdf"
    _write_synthetic_pdf(pdf_path, page_count=3, include_image=True)
    monkeypatch.setenv("DOC_STORE_PATH", str(tmp_path / "doc_store"))

    rendered = render_pdf_pages(str(pdf_path), "report/2025", scale=1)

    assert [page["page_number"] for page in rendered] == [1, 2, 3]
    assert all(page["page_width"] == 300 for page in rendered)
    assert all(page["page_height"] == 400 for page in rendered)
    assert all(page["pixel_width"] == 300 for page in rendered)
    assert all(page["pixel_height"] == 400 for page in rendered)
    assert rendered[0]["image_path"].startswith(
        f"assets/{document_asset_key('report/2025')}/pages/"
    )
    first_image_path = resolve_asset_path(rendered[0]["image_path"])
    assert first_image_path.is_file()
    with Image.open(first_image_path) as image:
        assert image.getbbox() == (0, 0, 300, 400)
        assert image.getpixel((50, 250)) != (255, 255, 255)
    with Image.open(resolve_asset_path(rendered[2]["image_path"])) as image:
        assert image.getpixel((150, 200))[0] > image.getpixel((150, 200))[1]

    repeated = render_pdf_pages(str(pdf_path), "report/2025", scale=1)
    assert [page["image_path"] for page in repeated] == [
        page["image_path"] for page in rendered
    ]
    page_files = list(first_image_path.parent.glob("*.png"))
    assert len(page_files) == 3


def test_render_validates_page_selection_and_scale(tmp_path, monkeypatch):
    pdf_path = tmp_path / "two-pages.pdf"
    _write_synthetic_pdf(pdf_path)
    monkeypatch.setenv("DOC_STORE_PATH", str(tmp_path / "doc_store"))

    with pytest.raises(PageRenderError, match="1-2 page range"):
        render_pdf_pages(str(pdf_path), "doc", pages={3})
    with pytest.raises(ValueError, match="scale"):
        render_pdf_pages(str(pdf_path), "doc", scale=0)


def test_render_rejects_invalid_pdf(tmp_path, monkeypatch):
    pdf_path = tmp_path / "invalid.pdf"
    pdf_path.write_text("not a PDF", encoding="utf-8")
    monkeypatch.setenv("DOC_STORE_PATH", str(tmp_path / "doc_store"))

    with pytest.raises(PageRenderError, match="Could not render PDF pages"):
        render_pdf_pages(str(pdf_path), "doc")


def test_asset_references_reject_paths_outside_storage_root(tmp_path, monkeypatch):
    monkeypatch.setenv("DOC_STORE_PATH", str(tmp_path / "store"))

    with pytest.raises(ValueError, match="escapes the document storage root"):
        resolve_asset_path("../outside.png")
