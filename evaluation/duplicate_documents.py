"""Read-only report of duplicate source PDFs in the document manifests."""

from __future__ import annotations

import json
from collections import defaultdict

from retrieval.document_store import list_document_manifests


def find_duplicate_documents(manifests=None) -> list[dict[str, object]]:
    """Group manifests by source SHA-256; never inspect or modify stored files."""
    if manifests is None:
        manifests = list_document_manifests()
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for manifest in manifests:
        digest = manifest.get("source_content_hash")
        document_id = manifest.get("document_id")
        if not digest or not document_id:
            continue
        grouped[str(digest)].append({
            "document_id": str(document_id),
            "source_name": str(manifest.get("source_name") or ""),
            "page_count": manifest.get("page_count"),
        })
    return [
        {"source_content_hash": digest, "documents": sorted(items, key=lambda item: item["document_id"])}
        for digest, items in sorted(grouped.items())
        if len(items) > 1
    ]


def main() -> None:
    print(json.dumps({"read_only": True, "duplicate_sources": find_duplicate_documents()}, indent=2))


if __name__ == "__main__":
    main()
