"""Partition PDFs into typed elements with stable, source-linked metadata."""

from __future__ import annotations

import hashlib
import logging
import math
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from ingestion.assets import asset_reference, document_asset_key, storage_root


def _asset_directory(asset_identity: str) -> Path:
    return (
        storage_root()
        / "assets"
        / document_asset_key(asset_identity)
        / "extracted"
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as asset_file:
        for block in iter(lambda: asset_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonicalize_extracted_asset(
    image_path: str,
    image_output_dir: Path,
    store_root: Path,
    canonical_assets: dict[Path, tuple[str, str]],
) -> tuple[str, str]:
    source = Path(image_path).expanduser().resolve()
    cached = canonical_assets.get(source)
    if cached:
        return cached
    if not source.is_file():
        return asset_reference(source, store_root), ""

    content_hash = _file_sha256(source)
    if source.is_relative_to(image_output_dir):
        suffix = source.suffix.lower() or ".img"
        target = image_output_dir / f"{content_hash}{suffix}"
        if source != target:
            if target.exists():
                source.unlink()
            else:
                source.replace(target)
        reference = asset_reference(target, store_root)
    else:
        reference = asset_reference(source, store_root)
    canonical_assets[source] = (reference, content_hash)
    return reference, content_hash


def _coordinate_metadata(element_metadata: Any) -> dict[str, Any]:
    coordinates = getattr(element_metadata, "coordinates", None)
    if coordinates is None:
        return {}

    points = getattr(coordinates, "points", None)
    system = getattr(coordinates, "system", None)
    try:
        point_values = [(float(point[0]), float(point[1])) for point in points]
    except (TypeError, ValueError, IndexError):
        point_values = []

    coordinate_system: dict[str, Any] = {}
    for name in ("width", "height", "orientation", "name", "origin"):
        value = getattr(system, name, None)
        if value is not None:
            coordinate_system[name] = value

    metadata: dict[str, Any] = {"coordinates": coordinates.to_dict()}
    if coordinate_system:
        metadata["coordinate_system"] = coordinate_system
        width = coordinate_system.get("width")
        height = coordinate_system.get("height")
        if (
            isinstance(width, (int, float))
            and isinstance(height, (int, float))
            and math.isfinite(width)
            and math.isfinite(height)
            and width > 0
            and height > 0
        ):
            metadata["page_width"] = width
            metadata["page_height"] = height
            if point_values:
                left = min(point[0] for point in point_values)
                top = min(point[1] for point in point_values)
                right = max(point[0] for point in point_values)
                bottom = max(point[1] for point in point_values)
                if (
                    all(math.isfinite(value) for point in point_values for value in point)
                    and 0 <= left <= right <= width
                    and 0 <= top <= bottom <= height
                ):
                    metadata["bbox"] = [left, top, right, bottom]
    return metadata


def make_element_uuid(
    document_id: str,
    page_number: int | None,
    element_type: str,
    content: str,
    occurrence: int = 0,
) -> str:
    """Return a stable UUID for the same element in a document re-ingestion."""
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    identity = (
        f"paperlens:{document_id}:{page_number}:{element_type}:"
        f"{content_hash}:{occurrence}"
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, identity))


def partition_pdf(
    pdf_path: str,
    document_id: str | None = None,
) -> list[dict[str, Any]]:
    """Partition a PDF and preserve its one-based page and asset provenance."""
    from unstructured.documents.elements import Image, NarrativeText, Table, Title
    from unstructured.partition.pdf import partition_pdf as _partition

    source_path = Path(pdf_path).expanduser().resolve()
    store_root = storage_root()
    image_output_dir = _asset_directory(
        document_id or f"source:{source_path}"
    )
    image_output_dir.mkdir(parents=True, exist_ok=True)

    poppler_available = all(shutil.which(command) for command in ("pdftoppm", "pdfinfo"))
    strategy = "hi_res" if poppler_available else "fast"
    if not poppler_available:
        logging.getLogger(__name__).warning(
            "Poppler was not found on PATH; using Unstructured's fast strategy. "
            "Table HTML and extracted figure assets may be unavailable."
        )
    raw_elements = _partition(
        filename=str(source_path),
        strategy=strategy,
        extract_images_in_pdf=poppler_available,
        extract_image_block_output_dir=str(image_output_dir),
        infer_table_structure=poppler_available,
    )

    type_map = {
        Title: "Title",
        NarrativeText: "NarrativeText",
        Table: "Table",
        Image: "Image",
    }
    occurrences: Counter[tuple[int | None, str, str]] = Counter()
    canonical_assets: dict[Path, tuple[str, str]] = {}
    results: list[dict[str, Any]] = []

    for element in raw_elements:
        element_type = type_map.get(type(element))
        if element_type is None:
            continue

        text = getattr(element, "text", "") or ""
        element_metadata = getattr(element, "metadata", None)
        page_number = getattr(element_metadata, "page_number", None)
        if not isinstance(page_number, int) or page_number < 1:
            page_number = None

        metadata: dict[str, Any] = {
            "source": source_path.name,
            "source_file": source_path.name,
            "source_path": str(source_path),
            "page": page_number,
            "element_type": element_type,
            "extraction_method": f"unstructured_{strategy}",
        }
        if document_id:
            metadata["document_id"] = document_id
        metadata.update(_coordinate_metadata(element_metadata))

        image_path = getattr(element_metadata, "image_path", None)
        asset_hash = ""
        if image_path:
            metadata["image_path"], asset_hash = _canonicalize_extracted_asset(
                image_path,
                image_output_dir,
                store_root,
                canonical_assets,
            )
            metadata["original_asset_path"] = metadata["image_path"]

        text_as_html = getattr(element_metadata, "text_as_html", None)
        if text_as_html:
            metadata["text_as_html"] = text_as_html

        content_hash = hashlib.sha256(
            f"{element_type}\0{text}\0{text_as_html or ''}\0{asset_hash}".encode(
                "utf-8"
            )
        ).hexdigest()
        metadata["content_hash"] = content_hash

        occurrence_key = (page_number, element_type, content_hash)
        occurrence = occurrences[occurrence_key]
        occurrences[occurrence_key] += 1
        element_uuid = (
            make_element_uuid(
                document_id,
                page_number,
                element_type,
                content_hash,
                occurrence,
            )
            if document_id
            else str(uuid.uuid4())
        )
        metadata["element_id"] = element_uuid
        results.append(
            {
                "uuid": element_uuid,
                "type": element_type,
                "content": text,
                "metadata": metadata,
            }
        )

    return results
