"""
Document store for raw tables and images.

Stores and retrieves raw elements (full HTML table, image path)
keyed by UUID. Default backend is local filesystem.
Swap _save / _load for S3 calls in production.
"""

import json
from pathlib import Path
from typing import Any

from ingestion.assets import document_asset_key, resolve_asset_path, storage_root


def save_raw_element(doc: dict[str, Any]) -> None:
    """Persist a raw element to the document store."""
    root = storage_root()
    root.mkdir(parents=True, exist_ok=True)
    dest = root / f"{doc['uuid']}.json"
    with open(dest, "w", encoding="utf-8") as f:
        # Store everything except large binary fields
        payload = {k: v for k, v in doc.items() if k != "_binary"}
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_raw_element(doc_uuid: str) -> dict[str, Any] | None:
    """Retrieve a raw element by UUID."""
    path = storage_root() / f"{doc_uuid}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_document_manifest(document_id: str, manifest: dict[str, Any]) -> Path:
    """Persist page-to-source provenance separately from semantically indexed items."""
    manifest_dir = storage_root() / "documents" / document_asset_key(document_id)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    destination = manifest_dir / "manifest.json"
    temporary = destination.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)
    temporary.replace(destination)
    return destination


def load_document_manifest(document_id: str) -> dict[str, Any] | None:
    """Load the provenance manifest for a document, if present."""
    path = (
        storage_root()
        / "documents"
        / document_asset_key(document_id)
        / "manifest.json"
    )
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as manifest_file:
        return json.load(manifest_file)


def list_document_manifests() -> list[dict[str, Any]]:
    """Load all valid document manifests stored under the configured root."""
    documents_dir = storage_root() / "documents"
    if not documents_dir.is_dir():
        return []

    manifests: list[dict[str, Any]] = []
    for path in sorted(documents_dir.glob("*/manifest.json")):
        with path.open("r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        if isinstance(manifest, dict) and manifest.get("document_id"):
            manifests.append(manifest)
    return manifests


def load_page_elements(document_id: str, page_number: int) -> list[dict[str, Any]]:
    """Load stored original elements belonging to one document page."""
    return [
        element
        for element in load_document_elements(document_id)
        if element.get("metadata", {}).get("page") == page_number
    ]


def load_document_elements(document_id: str) -> list[dict[str, Any]]:
    """Load stored original elements belonging to one document."""
    root = storage_root()
    if not root.is_dir():
        return []

    elements: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        with path.open("r", encoding="utf-8") as element_file:
            element = json.load(element_file)
        metadata = element.get("metadata", {})
        if metadata.get("document_id") == document_id:
            elements.append(element)
    return elements


__all__ = [
    "list_document_manifests",
    "load_document_elements",
    "load_document_manifest",
    "load_page_elements",
    "load_raw_element",
    "resolve_asset_path",
    "save_document_manifest",
    "save_raw_element",
]
