"""Validation for explicitly annotated PaperLens evaluation datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class DatasetValidationError(ValueError):
    """Raised when an evaluation dataset is malformed or ambiguously labelled."""


def validate_dataset(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise DatasetValidationError("Dataset must be an object with schema_version 1.")
    cases = value.get("cases")
    if not isinstance(cases, list):
        raise DatasetValidationError("Dataset cases must be a list.")
    if not isinstance(value.get("dataset_id"), str) or not value["dataset_id"].strip():
        raise DatasetValidationError("Dataset dataset_id is required.")
    seen: set[str] = set()
    for index, case in enumerate(cases):
        prefix = f"cases[{index}]"
        if not isinstance(case, dict):
            raise DatasetValidationError(f"{prefix} must be an object.")
        for key in ("case_id", "split", "category", "query", "source"):
            if not case.get(key):
                raise DatasetValidationError(f"{prefix}.{key} is required.")
        case_id = str(case["case_id"])
        if case_id in seen:
            raise DatasetValidationError(f"Duplicate case_id {case_id!r}.")
        seen.add(case_id)
        if not isinstance(case["source"], dict) or not case["source"].get("kind"):
            raise DatasetValidationError(f"{prefix}.source.kind is required.")
        if case["source"]["kind"] not in {
            "real_document", "synthetic_fixture", "mocked_result"
        }:
            raise DatasetValidationError(f"{prefix}.source.kind is unsupported.")
        if not case["source"].get("annotation_status"):
            raise DatasetValidationError(f"{prefix}.source.annotation_status is required.")
        if (
            case.get("retrieval_evaluation")
            and case["source"]["kind"] == "real_document"
            and not case["source"].get("document_id")
        ):
            raise DatasetValidationError(
                f"{prefix}.source.document_id is required for real retrieval cases."
            )
        if not isinstance(case["query"], str) or not case["query"].strip():
            raise DatasetValidationError(f"{prefix}.query must be non-empty text.")
        relevance = case.get("relevant_evidence", [])
        if not isinstance(relevance, list):
            raise DatasetValidationError(f"{prefix}.relevant_evidence must be a list.")
        identities: set[str] = set()
        for label_index, label in enumerate(relevance):
            if not isinstance(label, dict) or not label.get("id"):
                raise DatasetValidationError(
                    f"{prefix}.relevant_evidence[{label_index}].id is required."
                )
            if label.get("granularity") not in {"page", "figure", "element", "region", "chunk"}:
                raise DatasetValidationError(
                    f"{prefix}.relevant_evidence[{label_index}].granularity is invalid."
                )
            if not label.get("annotation"):
                raise DatasetValidationError(
                    f"{prefix}.relevant_evidence[{label_index}].annotation is required."
                )
            grade = label.get("relevance", 1)
            if type(grade) is not int or grade < 0:
                raise DatasetValidationError("Relevance grades must be non-negative integers.")
            if str(label["id"]) in identities:
                raise DatasetValidationError(f"Duplicate evidence label in {prefix}.")
            identities.add(str(label["id"]))
    return value


def load_dataset(path: str | Path) -> dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(f"Could not read dataset: {exc}") from exc
    return validate_dataset(value)
