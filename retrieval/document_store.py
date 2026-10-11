"""
Document store for raw tables and images.

Stores and retrieves raw elements (full HTML table, image path)
keyed by UUID. Default backend is local filesystem.
Swap _save / _load for S3 calls in production.
"""

import json
import math
import numbers
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from pathlib import Path, PurePath
from uuid import UUID
from typing import Any

from ingestion.assets import document_asset_key, resolve_asset_path, storage_root
from ingestion.structured_evidence import validate_structured_record


def _json_safe(value: Any, field_path: str = "$", *, _seen: set[int] | None = None) -> Any:
    """Convert metadata values to JSON types without discarding their meaning."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Enum):
        return _json_safe(value.value, field_path, _seen=_seen)
    if isinstance(value, PurePath):
        return value.as_posix()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (date, time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        # JSON has no decimal type; text preserves precision exactly.
        return str(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        converted = float(value)
        if not math.isfinite(converted):
            raise TypeError(f"Non-finite number at {field_path} cannot be represented as JSON.")
        return converted

    active = _seen if _seen is not None else set()
    identity = id(value)
    if identity in active:
        raise TypeError(f"Circular reference at {field_path} cannot be represented as JSON.")

    active.add(identity)
    try:
        if isinstance(value, Mapping):
            converted: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    if isinstance(key, Enum):
                        key = key.value
                    if isinstance(key, (str, int, float, bool, UUID, Path)):
                        key = str(key)
                    else:
                        raise TypeError(
                            f"Unsupported mapping key at {field_path}: "
                            f"{type(key).__module__}.{type(key).__qualname__}."
                        )
                if key in converted:
                    raise TypeError(f"Mapping keys collide after JSON conversion at {field_path}.{key}.")
                converted[key] = _json_safe(item, f"{field_path}.{key}", _seen=active)
            return converted
        if isinstance(value, (list, tuple)):
            return [
                _json_safe(item, f"{field_path}[{index}]", _seen=active)
                for index, item in enumerate(value)
            ]

        # Support Pydantic v2 and v1 without taking a dependency on either API.
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            try:
                model_value = model_dump(mode="python")
            except TypeError:
                model_value = model_dump()
            return _json_safe(model_value, field_path, _seen=active)
        model_dict = getattr(value, "dict", None)
        if callable(model_dict) and hasattr(value.__class__, "__fields__"):
            return _json_safe(model_dict(), field_path, _seen=active)
        if is_dataclass(value) and not isinstance(value, type):
            return _json_safe(asdict(value), field_path, _seen=active)

        # NumPy scalar values expose item(); arrays and arbitrary objects do not.
        if value.__class__.__module__.split(".", 1)[0] == "numpy":
            item = getattr(value, "item", None)
            if callable(item):
                return _json_safe(item(), field_path, _seen=active)

        raise TypeError(
            f"Unsupported value at {field_path}: "
            f"{type(value).__module__}.{type(value).__qualname__} cannot be represented as JSON."
        )
    finally:
        active.remove(identity)


def save_raw_element(doc: dict[str, Any]) -> None:
    """Persist a raw element to the document store."""
    root = storage_root()
    root.mkdir(parents=True, exist_ok=True)
    dest = root / f"{doc['uuid']}.json"
    with open(dest, "w", encoding="utf-8") as f:
        # Store everything except large binary fields
        payload = {k: v for k, v in doc.items() if k != "_binary"}
        json.dump(_json_safe(payload), f, ensure_ascii=False, indent=2, allow_nan=False)


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


def load_structured_evidence(document_id: str) -> list[dict[str, Any]]:
    """Load and validate the structured evidence embedded in one manifest."""
    manifest = load_document_manifest(document_id)
    if manifest is None:
        return []
    records = manifest.get("structured_evidence", [])
    if not isinstance(records, list):
        raise ValueError("Document manifest structured_evidence must be a list.")
    validated = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Document manifest contains a malformed structured record.")
        validated_record = validate_structured_record(record)
        if validated_record["document_id"] != document_id:
            raise ValueError("Structured evidence belongs to another document.")
        validated.append(validated_record)
    return validated


def save_structured_evidence(
    document_id: str,
    records: list[dict[str, Any]],
) -> int:
    """Idempotently replace structured records by their deterministic record IDs."""
    manifest = load_document_manifest(document_id)
    if manifest is None:
        raise ValueError(
            f"Document {document_id!r} has no manifest for structured evidence."
        )
    by_id: dict[str, dict[str, Any]] = {}
    existing = manifest.get("structured_evidence", [])
    if not isinstance(existing, list):
        raise ValueError("Document manifest structured_evidence must be a list.")
    for record in existing:
        if not isinstance(record, dict):
            raise ValueError("Document manifest contains a malformed structured record.")
        validated = validate_structured_record(record)
        if validated["document_id"] != document_id:
            raise ValueError("Structured evidence belongs to another document.")
        by_id[validated["record_id"]] = validated
    for record in records:
        validated = validate_structured_record(record)
        if validated["document_id"] != document_id:
            raise ValueError("Cannot persist evidence for a different document.")
        by_id[validated["record_id"]] = validated
    manifest["structured_evidence"] = list(by_id.values())
    save_document_manifest(document_id, manifest)
    return len(records)


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
    "load_structured_evidence",
    "resolve_asset_path",
    "save_document_manifest",
    "save_raw_element",
    "save_structured_evidence",
]
