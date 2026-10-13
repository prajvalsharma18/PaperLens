"""Conservative retrieval deduplication backed by source and content hashes."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def attach_source_hashes(
    items: list[dict[str, Any]],
    source_hashes: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Attach manifest SHA-256 provenance without changing original identity."""
    enriched = []
    for item in items:
        metadata = dict(item.get("metadata", {}))
        document_id = str(metadata.get("document_id", ""))
        source_hash = metadata.get("source_content_hash") or source_hashes.get(document_id)
        if source_hash:
            metadata["source_content_hash"] = str(source_hash)
        enriched.append({**item, "metadata": metadata})
    return enriched


def _dedup_key(item: dict[str, Any]) -> tuple[Any, ...] | None:
    metadata = item.get("metadata", {})
    source_hash = metadata.get("source_content_hash")
    content_hash = metadata.get("content_hash")
    if not source_hash or not content_hash:
        return None
    page = metadata.get("page", metadata.get("page_number"))
    kind = metadata.get("chunk_type") or item.get("type") or metadata.get("element_type")
    bbox = metadata.get("bbox")
    try:
        bbox_key = json.dumps(bbox, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        bbox_key = repr(bbox)
    return str(source_hash), page, str(kind), str(content_hash), bbox_key


def deduplicate_evidence(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse exact repeated evidence from byte-identical PDFs.

    Filename equality is deliberately irrelevant. Without both the original
    PDF hash and an element/asset content hash, candidates remain distinct.
    """
    unique: list[dict[str, Any]] = []
    positions: dict[tuple[Any, ...], int] = {}
    for item in items:
        key = _dedup_key(item)
        if key is None or key not in positions:
            if key is not None:
                positions[key] = len(unique)
            unique.append(item)
            continue

        position = positions[key]
        selected = unique[position]
        previous_metadata = dict(selected.get("metadata", {}))
        selected_metadata = dict(previous_metadata)
        duplicate_metadata = item.get("metadata", {})
        selected_document = str(selected_metadata.get("document_id", ""))
        duplicate_document = str(duplicate_metadata.get("document_id", ""))
        if duplicate_document and (
            not selected_document or duplicate_document < selected_document
        ):
            selected = item
            selected_metadata = dict(duplicate_metadata)
        document_ids = set()
        for metadata in (previous_metadata, duplicate_metadata):
            document_id = metadata.get("document_id")
            if document_id:
                document_ids.add(str(document_id))
            document_ids.update(str(value) for value in metadata.get("duplicate_document_ids", []))
        selected_metadata["duplicate_document_ids"] = sorted(document_ids)
        selected_metadata["duplicate_result_count"] = int(
            previous_metadata.get("duplicate_result_count", 1)
        ) + int(duplicate_metadata.get("duplicate_result_count", 1))

        selected_sources = dict(selected.get("retrieval_sources", {}))
        selected_sources.update(item.get("retrieval_sources", {}))
        if selected_sources:
            selected["retrieval_sources"] = selected_sources
            selected_metadata["retrieval_sources"] = selected_sources
        unique[position] = {**selected, "metadata": selected_metadata}
    return unique
