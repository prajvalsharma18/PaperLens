"""Versioned, provenance-preserving records for table and chart extraction."""

from __future__ import annotations

import hashlib
import json
from typing import Any


SCHEMA_VERSION = 1
VALID_STATUSES = {"verified", "estimated", "incomplete", "unreadable", "unsupported"}


def structured_record_id(
    document_id: str,
    source_element_id: str,
    kind: str,
) -> str:
    identity = f"paperlens:structured:{SCHEMA_VERSION}:{document_id}:{source_element_id}:{kind}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def validate_structured_record(record: dict[str, Any]) -> dict[str, Any]:
    required = (
        "record_id",
        "kind",
        "document_id",
        "source_element_id",
        "extraction_method",
        "status",
        "data",
    )
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"Structured record is missing required fields: {missing}.")
    if record["kind"] not in {"table", "chart"}:
        raise ValueError("Structured record kind must be 'table' or 'chart'.")
    if record["status"] not in VALID_STATUSES:
        raise ValueError(
            f"Structured record status must be one of {sorted(VALID_STATUSES)}."
        )
    if not isinstance(record["data"], dict):
        raise ValueError("Structured record data must be an object.")
    if not isinstance(record.get("warnings", []), list):
        raise ValueError("Structured record warnings must be a list.")
    if not isinstance(record.get("limitations", []), list):
        raise ValueError("Structured record limitations must be a list.")
    json.dumps(record, ensure_ascii=False, allow_nan=False)
    return record


def make_structured_record(
    *,
    kind: str,
    document_id: str,
    source_element_id: str,
    source_file: str | None,
    page: int | None,
    extraction_method: str,
    status: str,
    data: dict[str, Any],
    source_image_path: str | None = None,
    source_bbox: Any = None,
    region: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    limitations: list[str] | None = None,
    configuration_version: str = "1",
) -> dict[str, Any]:
    return validate_structured_record(
        {
            "schema_version": SCHEMA_VERSION,
            "record_id": structured_record_id(
                document_id, source_element_id, kind
            ),
            "kind": kind,
            "document_id": document_id,
            "source_element_id": source_element_id,
            "source_file": source_file,
            "page": page,
            "source_image_path": source_image_path,
            "source_bbox": source_bbox,
            "region": region,
            "extraction_method": extraction_method,
            "configuration_version": configuration_version,
            "status": status,
            "data": data,
            "warnings": warnings or [],
            "limitations": limitations or [],
        }
    )
