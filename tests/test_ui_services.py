import json
from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError

from ui import services


def test_pdf_validation_and_collision_resistant_ids():
    assert services.validate_pdf("notes.txt", b"%PDF-1.7", 1)
    assert services.validate_pdf("notes.pdf", b"not a pdf", 1)
    assert services.validate_pdf("notes.pdf", b"%PDF-1.7", 1) is None
    assert services.safe_document_id("A very useful paper.pdf") != services.safe_document_id(
        "A very useful paper.pdf"
    )


def test_upload_is_saved_in_document_scoped_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("DOC_STORE_PATH", str(tmp_path / "store"))
    document_id, destination = services.save_upload("notes.pdf", b"%PDF-1.7\n", 1)

    assert destination.is_file()
    assert destination.parent.name == document_id
    assert destination.read_bytes() == b"%PDF-1.7\n"


def test_document_discovery_distinguishes_manifest_and_pending_upload(tmp_path, monkeypatch):
    root = tmp_path / "store"
    monkeypatch.setenv("DOC_STORE_PATH", str(root))
    manifest_dir = root / "documents" / "stored-doc"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text(
        json.dumps({"document_id": "done-1", "source_file": "done.pdf", "page_count": 3}),
        encoding="utf-8",
    )
    pending_dir = root / "uploads" / "pending-1"
    pending_dir.mkdir(parents=True)
    (pending_dir / "pending.pdf").write_bytes(b"%PDF-1.7")

    documents = {item["document_id"]: item for item in services.list_documents()}

    assert documents["done-1"]["state"] == "Ingested"
    assert documents["done-1"]["page_count"] == 3
    assert documents["pending-1"]["state"] == "Uploaded · not ingested"


def test_run_cli_uses_argument_list_and_never_shell(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(services.subprocess, "run", fake_run)
    result = services.run_cli(["query", "question; not a shell command", "--document-id", "doc-1"])

    assert result.returncode == 0
    assert calls[0][0][1].endswith("main.py")
    assert calls[0][0][-3:] == ["question; not a shell command", "--document-id", "doc-1"]
    assert calls[0][1]["shell"] is False
    assert calls[0][1]["cwd"] == services.ROOT


def test_asset_resolution_accepts_document_assets_and_rejects_escape(tmp_path, monkeypatch):
    root = tmp_path / "store"
    monkeypatch.setenv("DOC_STORE_PATH", str(root))
    asset = root / "assets" / "page.png"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"image")

    assert services.resolve_asset("assets/page.png") == asset.resolve()
    assert services.resolve_asset("../outside.png") is None


def test_cli_output_redacts_configured_secret(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-value")

    assert services._redact("failure: test-secret-value") == "failure: [REDACTED]"


def test_query_output_preserves_answer_and_source_citations():
    answer, sources = services.parse_query_output(
        "Answer:\nThe report describes three phases.\n\nSources:\n- report.pdf, page 2 [E1]\n"
    )

    assert answer == "The report describes three phases."
    assert sources == ["report.pdf, page 2 [E1]"]


def test_failed_cli_result_is_returned_without_rewriting(monkeypatch):
    monkeypatch.setattr(
        services.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=2, stdout="partial", stderr="Qdrant unavailable"),
    )

    result = services.run_cli(["extract-data", "--document-id", "doc-1"])

    assert result.returncode == 2
    assert result.stdout == "partial"
    assert result.stderr == "Qdrant unavailable"


def test_qdrant_failure_is_reported_unavailable(monkeypatch):
    def unavailable(*args, **kwargs):
        raise URLError("offline")

    monkeypatch.setattr(services, "urlopen", unavailable)
    status = services.qdrant_status()

    assert status["reachable"] is False
    assert status["collections"] == []


def test_config_status_reports_flags_without_secret_values(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sensitive-test-key")
    status = services.config_status()

    assert status["OPENAI_API_KEY"] is True
    assert "sensitive-test-key" not in repr(status)
