"""Deterministic spatial tiling for large document image assets."""

from __future__ import annotations

import hashlib
import io
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from ingestion.assets import document_asset_key, resolve_asset_path, storage_root

CHUNKING_METHOD = "regular_grid_v1"


class VisualChunkError(ValueError):
    """Raised when visual chunk configuration or a source asset is invalid."""


@dataclass(frozen=True)
class VisualChunkingResult:
    chunks: list[dict[str, Any]]
    parent_width: int
    parent_height: int
    split: bool
    assets_reused: int


def validate_chunking_parameters(
    tile_width: int,
    tile_height: int,
    overlap: int,
    min_image_size: int,
) -> None:
    integer_settings = {
        "Tile width": tile_width,
        "Tile height": tile_height,
        "Tile overlap": overlap,
        "Minimum image size": min_image_size,
    }
    if any(not isinstance(value, int) or isinstance(value, bool) for value in integer_settings.values()):
        raise VisualChunkError("Tile dimensions, overlap, and minimum size must be integers.")
    if tile_width <= 0 or tile_height <= 0:
        raise VisualChunkError("Tile width and height must be positive integers.")
    if overlap < 0:
        raise VisualChunkError("Tile overlap cannot be negative.")
    if overlap >= min(tile_width, tile_height):
        raise VisualChunkError("Tile overlap must be smaller than both tile dimensions.")
    if min_image_size <= 0:
        raise VisualChunkError("Minimum image size must be a positive integer.")


def _tile_starts(length: int, tile_length: int, overlap: int) -> list[int]:
    if length <= tile_length:
        return [0]
    stride = tile_length - overlap
    starts: list[int] = []
    position = 0
    while position < length:
        starts.append(position)
        if position + tile_length >= length:
            break
        position += stride
    return starts


def _png_bytes(image: Image.Image) -> bytes:
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        normalized = background.convert("RGB")
        rgba.close()
        background.close()
    else:
        normalized = image.convert("RGB")

    buffer = io.BytesIO()
    try:
        normalized.save(buffer, format="PNG", optimize=False)
        return buffer.getvalue()
    finally:
        normalized.close()


def _write_content_addressed_asset(
    content: bytes,
    output_directory: Path,
) -> tuple[str, str, bool]:
    content_hash = hashlib.sha256(content).hexdigest()
    output_directory.mkdir(parents=True, exist_ok=True)
    destination = output_directory / f"{content_hash}.png"
    if destination.is_file():
        existing_hash = hashlib.sha256(destination.read_bytes()).hexdigest()
        if existing_hash == content_hash:
            return destination.relative_to(storage_root()).as_posix(), content_hash, True

    temporary = destination.with_name(f".{content_hash}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(content)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination.relative_to(storage_root()).as_posix(), content_hash, False


def chunk_image(
    parent_image: str | Path,
    parent_metadata: dict[str, Any],
    *,
    tile_width: int = 1024,
    tile_height: int = 1024,
    overlap: int = 128,
    min_image_size: int = 1024,
    output_directory: str | Path | None = None,
) -> VisualChunkingResult:
    """Create deterministic overlapping image tiles without semantic labels.

    Crop boxes use top-left-origin pixel coordinates and half-open bounds
    ``[x0, y0, x1, y1)`` relative to the immediate parent image.
    """
    validate_chunking_parameters(
        tile_width, tile_height, overlap, min_image_size
    )

    document_id = parent_metadata.get("document_id")
    parent_id = parent_metadata.get("element_id") or parent_metadata.get("uuid")
    if not document_id:
        raise VisualChunkError("Parent image metadata must include document_id.")
    if not parent_id:
        raise VisualChunkError("Parent image metadata must include an element ID.")

    parent_path = resolve_asset_path(parent_image)
    if not parent_path.is_file():
        raise VisualChunkError(f"Parent image does not exist: {parent_path}")

    try:
        with Image.open(parent_path) as parent:
            parent.load()
            parent_width, parent_height = parent.size
            needs_split = (
                parent_width > tile_width
                or parent_height > tile_height
                or max(parent_width, parent_height) > min_image_size
            )
            if not needs_split:
                return VisualChunkingResult(
                    chunks=[], parent_width=parent_width, parent_height=parent_height,
                    split=False, assets_reused=0,
                )

            if output_directory is None:
                asset_directory = (
                    storage_root()
                    / "assets"
                    / document_asset_key(str(document_id))
                    / "visual_chunks"
                )
            else:
                asset_directory = Path(output_directory).expanduser().resolve()
                try:
                    asset_directory.relative_to(storage_root())
                except ValueError as exc:
                    raise VisualChunkError(
                        "Visual chunk assets must be stored under DOC_STORE_PATH."
                    ) from exc

            x_starts = _tile_starts(parent_width, tile_width, overlap)
            y_starts = _tile_starts(parent_height, tile_height, overlap)
            chunks: list[dict[str, Any]] = []
            reused_count = 0
            parent_asset_path = str(parent_metadata.get("image_path") or parent_image)
            source_element_id = (
                parent_metadata.get("root_source_id")
                or parent_metadata.get("source_element_id")
                or parent_id
            )
            source_kind = parent_metadata.get("chunk_type")
            if source_kind not in ("figure", "page_image"):
                source_kind = (
                    "page_image"
                    if parent_metadata.get("extraction_method") == "pypdfium2_page_render"
                    else "figure"
                )

            for row, y0 in enumerate(y_starts):
                for column, x0 in enumerate(x_starts):
                    x1 = min(x0 + tile_width, parent_width)
                    y1 = min(y0 + tile_height, parent_height)
                    if x1 <= x0 or y1 <= y0:
                        raise VisualChunkError(
                            f"Invalid crop rectangle {(x0, y0, x1, y1)} for "
                            f"{parent_width}x{parent_height} parent."
                        )

                    crop = parent.crop((x0, y0, x1, y1))
                    try:
                        encoded = _png_bytes(crop)
                    finally:
                        crop.close()
                    asset_path, content_hash, reused = _write_content_addressed_asset(
                        encoded, asset_directory
                    )
                    reused_count += int(reused)
                    geometry = {
                        "bbox": [x0, y0, x1, y1],
                        "tile_width": tile_width,
                        "tile_height": tile_height,
                        "overlap": overlap,
                        "min_image_size": min_image_size,
                    }
                    identity = (
                        f"paperlens:{CHUNKING_METHOD}:{document_id}:{parent_id}:"
                        f"{content_hash}:{geometry}"
                    )
                    chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, identity))
                    metadata = {
                        "document_id": document_id,
                        "element_id": chunk_id,
                        "parent_id": parent_id,
                        "root_source_id": source_element_id,
                        "source_element_id": source_element_id,
                        "source_file": parent_metadata.get(
                            "source_file", parent_metadata.get("source")
                        ),
                        "source_path": parent_metadata.get("source_path"),
                        "page": parent_metadata.get("page"),
                        "chunk_type": "image_tile",
                        "parent_type": source_kind,
                        "parent_image_path": parent_asset_path,
                        "child_image_path": asset_path,
                        "bbox": [x0, y0, x1, y1],
                        "bbox_coordinate_system": {
                            "origin": "top-left",
                            "units": "pixels",
                            "convention": "half-open",
                            "relative_to": "parent_image",
                        },
                        "parent_width": parent_width,
                        "parent_height": parent_height,
                        "crop_width": x1 - x0,
                        "crop_height": y1 - y0,
                        "chunking_method": CHUNKING_METHOD,
                        "chunking_parameters": {
                            "tile_width": tile_width,
                            "tile_height": tile_height,
                            "overlap": overlap,
                            "min_image_size": min_image_size,
                        },
                        "tile_row": row,
                        "tile_column": column,
                        "content_hash": content_hash,
                        "extraction_method": parent_metadata.get(
                            "extraction_method"
                        ),
                        "source_bbox": parent_metadata.get("bbox"),
                        "source_coordinate_system": parent_metadata.get(
                            "coordinate_system"
                        ),
                    }
                    chunks.append(
                        {
                            "uuid": chunk_id,
                            "type": "ImageTile",
                            "content": "",
                            "metadata": metadata,
                        }
                    )

            return VisualChunkingResult(
                chunks=chunks,
                parent_width=parent_width,
                parent_height=parent_height,
                split=True,
                assets_reused=reused_count,
            )
    except (OSError, UnidentifiedImageError) as exc:
        raise VisualChunkError(
            f"Could not read parent image {parent_path}: {exc}"
        ) from exc
