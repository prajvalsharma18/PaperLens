"""
Unit tests for the PDF partitioner.
"""

from unittest.mock import patch
from pathlib import Path


def test_partition_returns_typed_elements(tmp_path, monkeypatch):
    """
    Partitioner should return dicts with 'type', 'content', 'uuid', 'metadata'.
    """
    from ingestion.partitioner import partition_pdf
    from unstructured.documents.elements import (
        CoordinateSystem,
        CoordinatesMetadata,
        ElementMetadata,
        NarrativeText,
        Table,
        Title,
    )
    from ingestion.assets import document_asset_key

    monkeypatch.setenv("DOC_STORE_PATH", str(tmp_path / "doc_store"))
    coordinates = CoordinatesMetadata(
        points=((0, 0), (1, 0), (1, 1)),
        system=CoordinateSystem(width=100, height=200),
    )
    mock_title = Title(
        text="Annual Report 2024",
        metadata=ElementMetadata(page_number=1, coordinates=coordinates),
    )
    mock_text = NarrativeText(
        text="Revenue increased by 15% in Q3.",
        metadata=ElementMetadata(page_number=2, image_path="figure.png"),
    )
    mock_table = Table(
        text="Revenue table",
        metadata=ElementMetadata(
            page_number=3,
            text_as_html="<table><tr><td>Q3</td></tr></table>",
        ),
    )

    with patch(
        "unstructured.partition.pdf.partition_pdf",
        return_value=[mock_title, mock_text, mock_table],
    ) as partition, patch(
        "ingestion.partitioner.shutil.which", return_value="mock-poppler-tool"
    ):
        results = partition_pdf("dummy.pdf")

    partition.assert_called_once_with(
        filename=str(Path("dummy.pdf").resolve()),
        strategy="hi_res",
        extract_images_in_pdf=True,
        extract_image_block_output_dir=str(
            (
                tmp_path
                / "doc_store"
                / "assets"
                / document_asset_key(
                    f"source:{Path('dummy.pdf').resolve()}"
                )
                / "extracted"
            ).resolve()
        ),
        infer_table_structure=True,
    )
    assert len(results) == 3
    assert results[0]["type"] == "Title"
    assert results[1]["type"] == "NarrativeText"
    assert results[2]["type"] == "Table"
    assert results[0]["content"] == "Annual Report 2024"
    assert results[1]["content"] == "Revenue increased by 15% in Q3."
    assert all("uuid" in r for r in results)
    assert results[0]["uuid"] != results[1]["uuid"]
    assert all("metadata" in r for r in results)
    assert results[0]["metadata"]["source"] == "dummy.pdf"
    assert results[0]["metadata"]["page"] == 1
    assert results[0]["metadata"]["coordinates"] == coordinates.to_dict()
    assert results[0]["metadata"]["bbox"] == [0.0, 0.0, 1.0, 1.0]
    assert results[0]["metadata"]["coordinate_system"]["width"] == 100
    assert results[0]["metadata"]["page_width"] == 100
    assert results[1]["metadata"]["page"] == 2
    assert results[1]["metadata"]["image_path"] == str(
        Path("figure.png").resolve()
    )
    assert results[2]["metadata"]["text_as_html"] == (
        "<table><tr><td>Q3</td></tr></table>"
    )


def test_partition_uses_fast_text_strategy_without_poppler(monkeypatch, tmp_path):
    from ingestion.partitioner import partition_pdf
    from ingestion.assets import document_asset_key

    monkeypatch.setenv("DOC_STORE_PATH", str(tmp_path / "store"))
    with patch(
        "unstructured.partition.pdf.partition_pdf", return_value=[]
    ) as partition, patch("ingestion.partitioner.shutil.which", return_value=None):
        assert partition_pdf("report.pdf", document_id="report-id") == []

    partition.assert_called_once_with(
        filename=str(Path("report.pdf").resolve()),
        strategy="fast",
        extract_images_in_pdf=False,
        extract_image_block_output_dir=str(
            (
                tmp_path
                / "store"
                / "assets"
                / document_asset_key("report-id")
                / "extracted"
            ).resolve()
        ),
        infer_table_structure=False,
    )


def test_partition_generates_stable_uuids_for_same_document(
    tmp_path, monkeypatch
):
    from ingestion.partitioner import partition_pdf
    from unstructured.documents.elements import ElementMetadata, NarrativeText

    monkeypatch.setenv("DOC_STORE_PATH", str(tmp_path / "doc_store"))
    element = NarrativeText(
        text="Repeatable page text",
        metadata=ElementMetadata(page_number=1),
    )
    with patch(
        "unstructured.partition.pdf.partition_pdf", return_value=[element]
    ), patch("ingestion.partitioner.shutil.which", return_value=None):
        first = partition_pdf("report.pdf", document_id="stable-id")
        second = partition_pdf("report.pdf", document_id="stable-id")

    assert first[0]["uuid"] == second[0]["uuid"]
    assert first[0]["metadata"]["content_hash"] == second[0]["metadata"]["content_hash"]


def test_extracted_image_assets_are_content_addressed_on_reingestion(
    tmp_path, monkeypatch
):
    from ingestion.assets import document_asset_key, resolve_asset_path
    from ingestion.partitioner import partition_pdf
    from unstructured.documents.elements import ElementMetadata, Image

    document_id = "stable-image-document"
    store = tmp_path / "doc_store"
    image_dir = (
        store / "assets" / document_asset_key(document_id) / "extracted"
    )
    image_dir.mkdir(parents=True)
    generated_image = image_dir / "figure.png"
    monkeypatch.setenv("DOC_STORE_PATH", str(store))

    def partition(*args, **kwargs):
        generated_image.write_bytes(b"same image bytes")
        return [
            Image(
                text="",
                metadata=ElementMetadata(
                    page_number=1,
                    image_path=str(generated_image),
                ),
            )
        ]

    with patch(
        "unstructured.partition.pdf.partition_pdf", side_effect=partition
    ), patch("ingestion.partitioner.shutil.which", return_value=None):
        first = partition_pdf("report.pdf", document_id=document_id)
        second = partition_pdf("report.pdf", document_id=document_id)

    assert first[0]["uuid"] == second[0]["uuid"]
    assert first[0]["metadata"]["image_path"] == second[0]["metadata"]["image_path"]
    assert resolve_asset_path(first[0]["metadata"]["image_path"]).is_file()
    assert sorted(path.name for path in image_dir.iterdir()) == [
        first[0]["metadata"]["image_path"].split("/")[-1]
    ]
