"""Run local, provenance-aware extraction for a stored PaperLens document."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from ingestion.assets import resolve_asset_path
from ingestion.chart_extractor import extract_chart
from ingestion.ocr import OcrError, ocr_image
from ingestion.structured_evidence import validate_structured_record
from ingestion.table_extractor import extract_table, extract_table_image
from retrieval.document_store import (
    load_document_elements,
    load_document_manifest,
    save_structured_evidence,
)


@dataclass(frozen=True)
class ExtractionReport:
    records: list[dict[str, Any]]
    status_counts: dict[str, int]
    failures: list[str]


def extract_document_data(document_id: str) -> ExtractionReport:
    """Extract stored tables and chart candidates without model or Qdrant calls."""
    if not document_id.strip():
        raise ValueError("Document ID cannot be empty.")
    manifest = load_document_manifest(document_id)
    if manifest is None:
        raise ValueError(f"Document {document_id!r} has no manifest.")
    if manifest.get("document_id") != document_id:
        raise ValueError("The document manifest ID does not match the requested ID.")

    records: list[dict[str, Any]] = []
    failures: list[str] = []
    elements = load_document_elements(document_id)
    if not elements:
        raise ValueError(f"Document {document_id!r} has no stored source elements.")
    for element in elements:
        metadata = element.get("metadata", {})
        try:
            if element.get("type") == "Table":
                records.append(extract_table(element, document_id=document_id))
            elif element.get("type") == "Image" and metadata.get(
                "structured_kind"
            ) == "table":
                try:
                    result = ocr_image(
                        str(resolve_asset_path(metadata["image_path"])),
                        int(metadata.get("page") or 0),
                    )
                except (OcrError, KeyError, OSError, ValueError) as exc:
                    records.append(
                        extract_table_image(
                            element,
                            document_id=document_id,
                            ocr_error=str(exc),
                        )
                    )
                else:
                    records.append(
                        extract_table_image(
                            element,
                            document_id=document_id,
                            ocr_result=result,
                        )
                    )
            elif element.get("type") == "Image" and metadata.get("chart_type"):
                records.append(extract_chart(element, document_id=document_id))
        except (OSError, ValueError) as exc:
            identity = element.get("uuid", "unknown element")
            failures.append(f"{identity}: {exc}")

    chunks = manifest.get("visual_chunks", [])
    if isinstance(chunks, list):
        for tile in chunks:
            if not isinstance(tile, dict):
                failures.append("Manifest contains a malformed visual chunk.")
                continue
            tile_metadata = tile.get("metadata", {})
            if tile_metadata.get("chart_type") is None and not tile_metadata.get(
                "structured_kind"
            ):
                continue
            element_id = tile.get("uuid") or tile_metadata.get("element_id")
            try:
                if tile_metadata.get("structured_kind") == "table":
                    failures.append(
                        f"{element_id or 'unknown tile'}: table-cell segmentation "
                        "from a crop is not implemented."
                    )
                    continue
                if not tile_metadata.get("chart_type"):
                    continue
                tile_element = {
                    "uuid": element_id,
                    "type": "Image",
                    "metadata": {
                        **tile_metadata,
                        "document_id": document_id,
                        "image_path": tile_metadata.get("child_image_path"),
                        "page": tile_metadata.get("page"),
                        "parent_id": tile_metadata.get("parent_id"),
                        "parent_image_path": tile_metadata.get("parent_image_path"),
                        "bbox": tile_metadata.get("bbox"),
                    },
                }
                records.append(extract_chart(tile_element, document_id=document_id))
            except (OSError, ValueError) as exc:
                failures.append(f"{element_id or 'unknown tile'}: {exc}")

    for record in records:
        validate_structured_record(record)
    save_structured_evidence(document_id, records)
    counts = Counter(record["status"] for record in records)
    return ExtractionReport(records, dict(counts), failures)
