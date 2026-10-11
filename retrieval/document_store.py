"""
Document store for raw tables and images.

Stores and retrieves raw elements (full HTML table, image path)
keyed by UUID. Default backend is local filesystem.
Swap _save / _load for S3 calls in production.
"""

import json
import os
from pathlib import Path
from typing import Any

DOC_STORE_PATH = Path(os.getenv("DOC_STORE_PATH", "./doc_store"))


def save_raw_element(doc: dict[str, Any]) -> None:
    """Persist a raw element to the document store."""
    DOC_STORE_PATH.mkdir(parents=True, exist_ok=True)
    dest = DOC_STORE_PATH / f"{doc['uuid']}.json"
    with open(dest, "w", encoding="utf-8") as f:
        # Store everything except large binary fields
        payload = {k: v for k, v in doc.items() if k != "_binary"}
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_raw_element(doc_uuid: str) -> dict[str, Any] | None:
    """Retrieve a raw element by UUID."""
    path = DOC_STORE_PATH / f"{doc_uuid}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
