"""Path helpers for document-scoped assets and safe asset references."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path


def storage_root() -> Path:
    return Path(os.getenv("DOC_STORE_PATH", "./doc_store")).expanduser().resolve()


def document_asset_key(document_id: str) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", document_id).strip(".-")[:48]
    digest = hashlib.sha256(document_id.encode("utf-8")).hexdigest()[:12]
    return f"{safe_id or 'document'}-{digest}"


def asset_reference(path: str | Path, root: Path | None = None) -> str:
    storage = (root or storage_root()).resolve()
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(storage).as_posix()
    except ValueError:
        return str(resolved)


def resolve_asset_path(path: str | Path, root: Path | None = None) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    storage = (root or storage_root()).resolve()
    resolved = (storage / candidate).resolve()
    if not resolved.is_relative_to(storage):
        raise ValueError(f"Asset path escapes the document storage root: {path}")
    return resolved
