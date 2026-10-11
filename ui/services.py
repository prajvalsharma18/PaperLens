"""Read-only workspace discovery and safe CLI operations for the UI."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def parse_query_output(output: str) -> tuple[str, list[str]]:
    """Split the answer and source section emitted by the existing CLI."""
    match = re.search(r"(?:^|\n)Answer:\s*\n?", output)
    if not match:
        return output.strip() or "The CLI returned no answer text.", []
    body = output[match.end():]
    source_match = re.search(r"\nSources:\s*\n", body)
    answer = (body[:source_match.start()] if source_match else body).strip()
    sources = []
    if source_match:
        sources = [line[2:].strip() for line in body[source_match.end():].splitlines() if line.startswith("- ")]
    return answer, sources


def storage_root() -> Path:
    configured = Path(os.getenv("DOC_STORE_PATH", "./doc_store")).expanduser()
    return (ROOT / configured).resolve() if not configured.is_absolute() else configured.resolve()


def safe_document_id(filename: str) -> str:
    stem = Path(filename).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")[:36] or "document"
    return f"{slug}-{uuid.uuid4().hex[:12]}"


def validate_pdf(filename: str, content: bytes, max_size_mb: int) -> str | None:
    if Path(filename).suffix.lower() != ".pdf":
        return "Choose a PDF file."
    if not content:
        return "The selected file is empty."
    if b"%PDF-" not in content[:1024]:
        return "The file does not contain a valid PDF header."
    if len(content) > max_size_mb * 1024 * 1024:
        return f"The file exceeds the {max_size_mb} MB upload limit."
    return None


def save_upload(filename: str, content: bytes, max_size_mb: int) -> tuple[str, Path]:
    error = validate_pdf(filename, content, max_size_mb)
    if error:
        raise ValueError(error)
    document_id = safe_document_id(filename)
    safe_name = Path(filename.replace("\\", "/")).name.replace("..", "_")
    destination_dir = storage_root() / "uploads" / document_id
    destination_dir.mkdir(parents=True, exist_ok=False)
    destination = destination_dir / safe_name
    destination.write_bytes(content)
    return document_id, destination


@dataclass(frozen=True)
class CLIResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


def _redact(text: str) -> str:
    for name in ("OPENAI_API_KEY", "QDRANT_API_KEY"):
        secret = os.getenv(name, "")
        if secret and secret not in {"your-openai-api-key-here"}:
            text = text.replace(secret, "[REDACTED]")
    return text


def run_cli(arguments: list[str], timeout: int | None = None) -> CLIResult:
    """Run one known PaperLens CLI action without a shell."""
    allowed = {"ingest", "query", "index-visuals", "extract-data"}
    if not arguments or arguments[0] not in allowed:
        raise ValueError("Unsupported PaperLens CLI operation.")
    seconds = timeout or int(os.getenv("PAPERLENS_CLI_TIMEOUT", "1800"))
    try:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "main.py"), *arguments],
            cwd=ROOT,
            shell=False,
            capture_output=True,
            text=True,
            timeout=seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        return CLIResult(-1, _redact(stdout), _redact(stderr), timed_out=True)
    return CLIResult(
        completed.returncode,
        _redact(completed.stdout),
        _redact(completed.stderr),
    )


def list_documents() -> list[dict[str, Any]]:
    """Discover manifest-backed documents and explicitly saved pending uploads."""
    root = storage_root()
    manifests: dict[str, dict[str, Any]] = {}
    manifest_dir = root / "documents"
    if manifest_dir.is_dir():
        for path in manifest_dir.glob("*/manifest.json"):
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            document_id = manifest.get("document_id") if isinstance(manifest, dict) else None
            if document_id:
                manifests[str(document_id)] = manifest

    documents = []
    for document_id, manifest in manifests.items():
        documents.append({
            "document_id": document_id,
            "filename": manifest.get("source_file") or "Unknown filename",
            "page_count": manifest.get("page_count"),
            "state": "Ingested",
            "manifest": manifest,
        })

    uploads_dir = root / "uploads"
    if uploads_dir.is_dir():
        for directory in uploads_dir.iterdir():
            if not directory.is_dir() or directory.name in manifests:
                continue
            candidate = next((p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"), None)
            if candidate:
                documents.append({
                    "document_id": directory.name,
                    "filename": candidate.name,
                    "page_count": None,
                    "state": "Uploaded · not ingested",
                    "manifest": None,
                })
    return sorted(documents, key=lambda item: item["filename"].lower())


def document_elements(document_id: str) -> list[dict[str, Any]]:
    elements = []
    root = storage_root()
    if not root.is_dir():
        return elements
    for path in root.glob("*.json"):
        try:
            element = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        metadata = element.get("metadata", {}) if isinstance(element, dict) else {}
        if isinstance(metadata, dict) and metadata.get("document_id") == document_id:
            elements.append(element)
    return elements


def resolve_asset(reference: str | None) -> Path | None:
    if not reference:
        return None
    root = storage_root()
    candidate = Path(reference).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        return None
    return resolved if resolved.is_file() else None


def qdrant_status(timeout: float = 2.0) -> dict[str, Any]:
    base = os.getenv("QDRANT_URL", "http://localhost:6333").rstrip("/")
    headers = {"api-key": os.getenv("QDRANT_API_KEY", "")} if os.getenv("QDRANT_API_KEY") else {}
    try:
        request = Request(f"{base}/collections", headers=headers)
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        names = [item.get("name") for item in payload.get("result", {}).get("collections", [])]
        return {"reachable": True, "collections": names, "error": None}
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        return {"reachable": False, "collections": [], "error": str(exc)}


def qdrant_collection_count(name: str, timeout: float = 2.0) -> int | None:
    base = os.getenv("QDRANT_URL", "http://localhost:6333").rstrip("/")
    headers = {"Content-Type": "application/json"}
    if os.getenv("QDRANT_API_KEY"):
        headers["api-key"] = os.environ["QDRANT_API_KEY"]
    request = Request(f"{base}/collections/{name}", headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        result = payload.get("result", {})
        return result.get("points_count", result.get("indexed_vectors_count"))
    except (HTTPError, URLError, TimeoutError, OSError, ValueError):
        return None


def qdrant_document_count(collection: str, document_id: str, timeout: float = 2.0) -> int | None:
    base = os.getenv("QDRANT_URL", "http://localhost:6333").rstrip("/")
    headers = {"Content-Type": "application/json"}
    if os.getenv("QDRANT_API_KEY"):
        headers["api-key"] = os.environ["QDRANT_API_KEY"]
    body = json.dumps({
        "filter": {"must": [{"key": "document_id", "match": {"value": document_id}}]},
        "exact": True,
    }).encode("utf-8")
    request = Request(f"{base}/collections/{collection}/points/count", data=body, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload.get("result", {}).get("count")
    except (HTTPError, URLError, TimeoutError, OSError, ValueError):
        return None


def config_status() -> dict[str, bool]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    return {
        "OPENAI_API_KEY": bool(api_key and api_key != "your-openai-api-key-here"),
        "QDRANT_URL": bool(os.getenv("QDRANT_URL", "http://localhost:6333").strip()),
        "DOC_STORE_PATH": bool(os.getenv("DOC_STORE_PATH", "./doc_store").strip()),
    }
