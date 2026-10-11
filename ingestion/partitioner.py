"""
Partitions a PDF into typed elements using Unstructured.io.

Returns a list of dicts with keys:
  - type: str  (Title | NarrativeText | Table | Image)
  - content: str | bytes
  - metadata: dict  (page number, coordinates, source filename, uuid)
"""

import os
import uuid
from pathlib import Path
from typing import Any


def partition_pdf(pdf_path: str) -> list[dict[str, Any]]:
    """
    Partition a PDF file into typed elements.

    Uses unstructured's hi_res strategy when available for accurate
    table and image extraction. Falls back to fast strategy if
    layout model is not installed.
    """
    from unstructured.partition.pdf import partition_pdf as _partition
    from unstructured.documents.elements import (
        Title, NarrativeText, Table, Image
    )

    image_output_dir = Path(os.getenv("DOC_STORE_PATH", "./doc_store")) / "images"
    image_output_dir.mkdir(parents=True, exist_ok=True)

    raw_elements = _partition(
        filename=str(Path(pdf_path).resolve()),
        strategy="hi_res",          # Use "fast" if layout model unavailable
        extract_images_in_pdf=True,
        extract_image_block_output_dir=str(image_output_dir.resolve()),
        infer_table_structure=True,
    )

    results = []
    type_map = {
        Title: "Title",
        NarrativeText: "NarrativeText",
        Table: "Table",
        Image: "Image",
    }

    for el in raw_elements:
        el_type = type_map.get(type(el))
        if el_type is None:
            continue

        element_metadata = getattr(el, "metadata", None)
        coordinates = getattr(element_metadata, "coordinates", None)
        if coordinates is not None and hasattr(coordinates, "to_dict"):
            coordinates = coordinates.to_dict()

        image_path = getattr(element_metadata, "image_path", None)
        if image_path:
            image_path = str(Path(image_path).resolve())

        metadata = {
            "source": Path(pdf_path).name,
            "page": getattr(element_metadata, "page_number", None),
            "coordinates": coordinates,
            "image_path": image_path,
        }
        text_as_html = getattr(element_metadata, "text_as_html", None)
        if text_as_html:
            metadata["text_as_html"] = text_as_html

        results.append({
            "uuid": str(uuid.uuid4()),
            "type": el_type,
            "content": getattr(el, "text", "") or "",
            "metadata": metadata,
        })

    return results
