"""Query-scoped evidence identities and user-facing source references."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def build_evidence_package(context: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy retrieved items and attach stable IDs without changing source metadata."""
    package: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in context:
        metadata = dict(item.get("metadata", {}))
        identity = str(
            item.get("uuid")
            or metadata.get("record_id")
            or metadata.get("element_id")
            or metadata.get("chunk_id")
            or hashlib.sha256(repr(item).encode("utf-8")).hexdigest()
        )
        if identity in seen:
            continue
        seen.add(identity)
        evidence_id = "E-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:10]
        metadata["evidence_id"] = evidence_id
        source = metadata.get("source") or metadata.get("source_file")
        document_id = metadata.get("document_id")
        page = metadata.get("page", metadata.get("page_number"))
        label = metadata.get("evidence_label")
        if not label:
            parts = [str(source or document_id or "Retrieved source")]
            if page is not None:
                parts.append(f"page {page}")
            element_id = item.get("uuid") or metadata.get("element_id")
            if element_id:
                parts.append(f"element {element_id}")
            label = ", ".join(parts)
        metadata["evidence_label"] = str(label)
        package.append({**item, "metadata": metadata})
    return package


def format_source_citations(context: list[dict[str, Any]]) -> list[str]:
    """Format references from evidence actually present in the query package."""
    citations: list[str] = []
    for item in context:
        metadata = item.get("metadata", {})
        label = metadata.get("evidence_label")
        if not label:
            continue
        evidence_id = metadata.get("evidence_id")
        source = metadata.get("source") or metadata.get("source_file")
        page = metadata.get("page", metadata.get("page_number"))
        identity = item.get("uuid") or metadata.get("element_id")
        if source and Path(str(source)).is_absolute():
            source = Path(str(source)).name
        details = [str(source)] if source else [str(label)]
        if page is not None and f"page {page}" not in details[0]:
            details.append(f"page {page}")
        document_id = metadata.get("document_id")
        if document_id:
            details.append(f"document ID: {document_id}")
        if identity:
            details.append(f"element {identity}")
        evidence_type = metadata.get("evidence_type")
        if evidence_type in {"visual_child_crop", "visual_parent_context"}:
            details.append(str(evidence_type).replace("_", " "))
        citation = ", ".join(details)
        if evidence_id:
            citation += f" [{evidence_id}]"
        if citation not in citations:
            citations.append(citation)
    return citations


def validate_answer_citations(answer: str, context: list[dict[str, Any]]) -> str:
    """Drop model citation tokens that do not identify supplied evidence."""
    import re

    valid = {
        item.get("metadata", {}).get("evidence_id")
        for item in context
        if item.get("metadata", {}).get("evidence_id")
    }
    return re.sub(
        r"\[(E-[0-9a-f]{10})\]",
        lambda match: match.group(0) if match.group(1) in valid else "",
        answer,
    )
