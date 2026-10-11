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
    ) as partition:
        results = partition_pdf("dummy.pdf")

    partition.assert_called_once_with(
        filename=str(Path("dummy.pdf").resolve()),
        strategy="hi_res",
        extract_images_in_pdf=True,
        extract_image_block_output_dir=str(
            (tmp_path / "doc_store" / "images").resolve()
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
    assert results[1]["metadata"]["page"] == 2
    assert results[1]["metadata"]["image_path"] == str(
        Path("figure.png").resolve()
    )
    assert results[2]["metadata"]["text_as_html"] == (
        "<table><tr><td>Q3</td></tr></table>"
    )
