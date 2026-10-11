"""Manifest-driven discovery, indexing, and searching of visual assets."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from ingestion.assets import resolve_asset_path
from retrieval.document_store import load_document_elements, load_document_manifest
from retrieval.visual_embedder import VisualEmbedder, get_visual_embedder
from retrieval.visual_vector_store import (
    VisualVectorStoreError,
    search_visual_vectors,
    upsert_visual_vectors,
    validate_visual_collection_dimension,
)


@dataclass(frozen=True)
class VisualIndexResult:
    parents_indexed: int
    page_images_indexed: int
    tiles_indexed: int
    skipped_assets: int
    skip_reasons: dict[str, int]
    model_name: str | None
    vector_dimension: int | None


class VisualIndexError(RuntimeError):
    """Raised when visual embeddings cannot be written consistently."""


def _valid_image_size(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.size
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"Image is unreadable: {exc}") from exc


def _parent_record(
    document_id: str,
    element: dict[str, Any],
) -> dict[str, Any]:
    metadata = element.get("metadata", {})
    element_id = element.get("uuid") or metadata.get("element_id")
    asset_path = metadata.get("image_path")
    if not element_id or not asset_path:
        raise ValueError("Image element has no stable ID or image asset path.")
    resolved_path = resolve_asset_path(asset_path)
    width, height = _valid_image_size(resolved_path)
    page_number = metadata.get("page")
    is_page = metadata.get("extraction_method") == "pypdfium2_page_render"
    chunk_type = "page_image" if is_page else "figure"
    return {
        "document_id": document_id,
        "element_id": str(element_id),
        "parent_id": str(element_id),
        "root_source_id": str(element_id),
        "source_file": metadata.get("source_file", metadata.get("source")),
        "page_number": page_number,
        "chunk_type": chunk_type,
        "asset_path": str(asset_path),
        "parent_asset_path": str(asset_path),
        "bbox": metadata.get("bbox"),
        "bbox_coordinate_system": metadata.get("coordinate_system"),
        "image_width": width,
        "image_height": height,
        "content_hash": metadata.get("content_hash"),
        "extraction_method": metadata.get("extraction_method"),
        "_resolved_path": str(resolved_path),
    }


def _tile_record(document_id: str, tile: dict[str, Any]) -> dict[str, Any]:
    metadata = tile.get("metadata", {})
    element_id = tile.get("uuid") or metadata.get("element_id")
    asset_path = metadata.get("child_image_path")
    parent_id = metadata.get("parent_id")
    if not element_id or not asset_path or not parent_id:
        raise ValueError("Visual tile lacks a stable ID, asset path, or parent ID.")
    resolved_path = resolve_asset_path(asset_path)
    width, height = _valid_image_size(resolved_path)
    parameters = metadata.get("chunking_parameters")
    if not isinstance(parameters, dict):
        parameters = {}
    return {
        "document_id": document_id,
        "element_id": str(element_id),
        "chunk_id": str(element_id),
        "parent_id": str(parent_id),
        "root_source_id": str(
            metadata.get("root_source_id")
            or metadata.get("source_element_id")
            or parent_id
        ),
        "source_file": metadata.get("source_file"),
        "page_number": metadata.get("page"),
        "chunk_type": "image_tile",
        "asset_path": str(asset_path),
        "parent_asset_path": metadata.get("parent_image_path"),
        "bbox": metadata.get("bbox"),
        "bbox_coordinate_system": metadata.get("bbox_coordinate_system"),
        "image_width": width,
        "image_height": height,
        "parent_width": metadata.get("parent_width"),
        "parent_height": metadata.get("parent_height"),
        "content_hash": metadata.get("content_hash"),
        "chunking_method": metadata.get("chunking_method"),
        "chunking_parameters": parameters,
        "tile_row": metadata.get("tile_row"),
        "tile_column": metadata.get("tile_column"),
        "source_bbox": metadata.get("source_bbox"),
        "source_coordinate_system": metadata.get("source_coordinate_system"),
        "_resolved_path": str(resolved_path),
    }


def discover_visual_assets(
    document_id: str,
    manifest: dict[str, Any],
    elements: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Discover extracted/opted-in page images and manifest child tiles."""
    records: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    seen: set[str] = set()
    for element in elements:
        if element.get("type") != "Image":
            continue
        try:
            record = _parent_record(document_id, element)
        except (OSError, ValueError) as exc:
            skipped[str(exc)] += 1
            continue
        identity = record["element_id"]
        if identity in seen:
            continue
        seen.add(identity)
        records.append(record)

    visual_chunks = manifest.get("visual_chunks", [])
    if not isinstance(visual_chunks, list):
        skipped["Manifest visual_chunks field is not a list."] += 1
        return records, skipped
    for tile in visual_chunks:
        if not isinstance(tile, dict):
            skipped["Manifest contains a malformed visual tile record."] += 1
            continue
        try:
            record = _tile_record(document_id, tile)
        except (OSError, ValueError) as exc:
            skipped[str(exc)] += 1
            continue
        identity = record["element_id"]
        if identity in seen:
            continue
        seen.add(identity)
        records.append(record)
    return records, skipped


def index_document_visual_assets(
    document_id: str,
    *,
    batch_size: int = 16,
    embedder: VisualEmbedder | Any | None = None,
    client: Any = None,
) -> VisualIndexResult:
    if batch_size < 1:
        raise ValueError("Visual indexing batch size must be positive.")
    manifest = load_document_manifest(document_id)
    if manifest is None:
        raise ValueError(
            f"Document {document_id!r} has no manifest. Ingest the PDF first."
        )
    if manifest.get("document_id") != document_id:
        raise ValueError("The document manifest ID does not match the requested ID.")

    records, skipped = discover_visual_assets(
        document_id,
        manifest,
        load_document_elements(document_id),
    )
    if not records:
        return VisualIndexResult(
            0, 0, 0, sum(skipped.values()), dict(skipped), None, None
        )

    model = embedder if embedder is not None else get_visual_embedder()
    paths = [record["_resolved_path"] for record in records]
    vectors = model.encode_images(paths, batch_size=batch_size)
    if len(vectors) != len(records):
        raise VisualIndexError(
            f"Visual embedder returned {len(vectors)} vectors for "
            f"{len(records)} assets."
        )
    payload_records = [
        {key: value for key, value in record.items() if key != "_resolved_path"}
        for record in records
    ]
    upsert_visual_vectors(
        payload_records,
        vectors,
        model.dimension,
        client=client,
    )
    parents_indexed = sum(
        record["chunk_type"] in {"figure", "page_image"} for record in records
    )
    page_images = sum(record["chunk_type"] == "page_image" for record in records)
    tiles = sum(record["chunk_type"] == "image_tile" for record in records)
    return VisualIndexResult(
        parents_indexed=parents_indexed,
        page_images_indexed=page_images,
        tiles_indexed=tiles,
        skipped_assets=sum(skipped.values()),
        skip_reasons=dict(skipped),
        model_name=getattr(model, "model_name", None),
        vector_dimension=model.dimension,
    )


def search_visual_assets(
    question: str,
    *,
    limit: int,
    document_id: str | None = None,
    embedder: VisualEmbedder | Any | None = None,
    client: Any = None,
) -> list[dict[str, Any]]:
    from retrieval.visual_vector_store import visual_collection_exists

    if not visual_collection_exists(client):
        return []
    model = embedder if embedder is not None else get_visual_embedder()
    if not validate_visual_collection_dimension(model.dimension, client):
        return []
    query_vector = model.encode_text(question)
    try:
        points = search_visual_vectors(
            query_vector,
            limit=limit,
            document_id=document_id,
            client=client,
        )
    except VisualVectorStoreError:
        raise
    results: list[dict[str, Any]] = []
    for point in points:
        payload = getattr(point, "payload", None)
        if not isinstance(payload, dict):
            continue
        asset_path = payload.get("asset_path")
        element_id = payload.get("element_id")
        if not asset_path or not element_id:
            continue
        try:
            resolved_path = resolve_asset_path(asset_path)
            if not resolved_path.is_file():
                continue
            _valid_image_size(resolved_path)
        except ValueError:
            continue
        results.append(
            {
                "uuid": str(element_id),
                "type": "Image",
                "content": "",
                "metadata": {
                    **payload,
                    "image_path": asset_path,
                    "page": payload.get("page_number"),
                    "evidence_type": payload.get("chunk_type"),
                },
                "visual_score": getattr(point, "score", None),
            }
        )
    return results
