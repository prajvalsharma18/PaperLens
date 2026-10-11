"""Tests for the PaperLens CLI and pipeline orchestration."""

from unittest.mock import MagicMock

import pytest

import main


def _write_pdf(tmp_path, name="sample.pdf", content=b"%PDF-1.7\n"):
    path = tmp_path / name
    path.write_bytes(content)
    return path


def _elements(image_path):
    return [
        {
            "uuid": "title-id",
            "type": "Title",
            "content": "Quarterly report",
            "metadata": {"source": "sample.pdf", "page": 1},
        },
        {
            "uuid": "table-id",
            "type": "Table",
            "content": "Revenue Q1 $10",
            "metadata": {
                "source": "sample.pdf",
                "page": 2,
                "text_as_html": "<table><tr><td>Q1</td><td>$10</td></tr></table>",
            },
        },
        {
            "uuid": "image-id",
            "type": "Image",
            "content": "",
            "metadata": {
                "source": "sample.pdf",
                "page": 3,
                "image_path": str(image_path),
            },
        },
    ]


@pytest.fixture
def valid_configuration(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.test:6333")
    monkeypatch.setenv("DOC_STORE_PATH", str(tmp_path / "doc_store"))


def test_cli_help_for_root_and_commands(capsys):
    for args in (["--help"], ["ingest", "--help"], ["query", "--help"]):
        with pytest.raises(SystemExit) as exit_info:
            main.main(args)
        assert exit_info.value.code == 0
        assert "usage:" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("filename", "content", "message"),
    [
        ("missing.pdf", b"", "does not exist"),
        ("report.txt", b"%PDF-1.7", ".pdf extension"),
        ("invalid.pdf", b"not a PDF", "PDF signature"),
    ],
)
def test_ingest_rejects_invalid_pdf_before_external_setup(
    tmp_path, monkeypatch, filename, content, message
):
    monkeypatch.setattr(main, "_validate_configuration", MagicMock())
    path = tmp_path / filename
    if filename != "missing.pdf":
        path.write_bytes(content)

    with pytest.raises(main.PaperLensError, match=message):
        main.ingest_pdf(str(path))

    main._validate_configuration.assert_not_called()


def test_ingest_rejects_missing_credentials_before_qdrant_or_paid_calls(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.test:6333")
    monkeypatch.setattr(main, "get_vector_store", MagicMock())
    monkeypatch.setattr(main, "partition_pdf", MagicMock())

    with pytest.raises(main.PaperLensError, match="OPENAI_API_KEY"):
        main.ingest_pdf(str(_write_pdf(tmp_path)))

    main.get_vector_store.assert_not_called()
    main.partition_pdf.assert_not_called()


def test_ingest_indexes_search_text_and_saves_same_original_elements(
    tmp_path, monkeypatch, capsys, valid_configuration
):
    pdf_path = _write_pdf(tmp_path)
    image_path = tmp_path / "chart.png"
    image_path.write_bytes(b"image")
    elements = _elements(image_path)
    embedder = MagicMock(name="embedder")
    saved = []
    indexed = {}
    monkeypatch.setattr(main, "get_embedder", lambda: embedder)
    monkeypatch.setattr(main, "get_vector_store", MagicMock())
    monkeypatch.setattr(main, "partition_pdf", lambda path: elements)
    monkeypatch.setattr(
        main,
        "summarise_images",
        lambda images: [
            {
                **images[0],
                "embed_content": "Figure description: revenue chart",
                "vision_summary": "Revenue rises in Q1.",
            }
        ],
    )
    monkeypatch.setattr(main, "save_raw_element", saved.append)

    def upsert(documents, received_embedder):
        indexed["documents"] = documents
        indexed["embedder"] = received_embedder

    monkeypatch.setattr(main, "upsert_documents", upsert)

    assert main.ingest_pdf(str(pdf_path), document_id="report-2025") == 3

    assert indexed["embedder"] is embedder
    assert [document["uuid"] for document in indexed["documents"]] == [
        "table-id",
        "image-id",
        "title-id",
    ]
    assert {doc["uuid"] for doc in saved} == {
        doc["uuid"] for doc in indexed["documents"]
    }
    assert all(doc["metadata"]["document_id"] == "report-2025" for doc in saved)
    assert all(
        doc["metadata"]["document_id"] == "report-2025"
        for doc in indexed["documents"]
    )
    table = next(doc for doc in indexed["documents"] if doc["uuid"] == "table-id")
    assert table["content"] == elements[1]["metadata"]["text_as_html"]
    assert table["embed_content"].startswith("Table content:")
    image = next(doc for doc in indexed["documents"] if doc["uuid"] == "image-id")
    assert image["embed_content"] == "Figure description: revenue chart"
    assert image["metadata"]["image_path"] == str(image_path)
    report = capsys.readouterr().out
    assert "Indexed elements: 3" in report
    assert "Document ID: report-2025" in report


def test_ingest_surfaces_partition_and_index_errors(tmp_path, monkeypatch, valid_configuration):
    monkeypatch.setattr(main, "get_embedder", lambda: MagicMock())
    monkeypatch.setattr(main, "get_vector_store", lambda embedder: object())
    monkeypatch.setattr(
        main, "partition_pdf", MagicMock(side_effect=RuntimeError("partition failure"))
    )

    with pytest.raises(main.PaperLensError, match="PDF parsing failed"):
        main.ingest_pdf(str(_write_pdf(tmp_path)))

    monkeypatch.setattr(
        main, "partition_pdf", lambda path: [
            {
                "uuid": "text-id",
                "type": "NarrativeText",
                "content": "Some content",
                "metadata": {"page": 1},
            }
        ],
    )
    monkeypatch.setattr(main, "save_raw_element", MagicMock())
    monkeypatch.setattr(
        main, "upsert_documents", MagicMock(side_effect=RuntimeError("index failure"))
    )

    with pytest.raises(main.PaperLensError, match="Embedding or Qdrant indexing failed"):
        main.ingest_pdf(str(_write_pdf(tmp_path)))


def test_ingest_counts_and_reports_unsupported_elements(
    tmp_path, monkeypatch, capsys, valid_configuration
):
    monkeypatch.setattr(main, "get_embedder", lambda: MagicMock())
    monkeypatch.setattr(main, "get_vector_store", lambda embedder: object())
    monkeypatch.setattr(
        main,
        "partition_pdf",
        lambda path: [
            {
                "uuid": "text-id",
                "type": "NarrativeText",
                "content": "A paragraph",
                "metadata": {"page": 1},
            },
            {
                "uuid": "empty-id",
                "type": "NarrativeText",
                "content": "",
                "metadata": {},
            },
            {
                "uuid": "unsupported-id",
                "type": "ListItem",
                "content": "An item",
                "metadata": {},
            },
            {
                "uuid": "image-id",
                "type": "Image",
                "content": "",
                "metadata": {"image_path": str(tmp_path / "missing.png")},
            },
        ],
    )
    saved = []
    indexed = MagicMock()
    monkeypatch.setattr(main, "save_raw_element", saved.append)
    monkeypatch.setattr(main, "upsert_documents", indexed)

    assert main.ingest_pdf(str(_write_pdf(tmp_path))) == 1

    assert len(saved) == 1
    indexed.assert_called_once()
    output = capsys.readouterr().out
    assert "Extracted elements: 4" in output
    assert "Skipped elements: 3" in output
    assert "empty text element: 1" in output
    assert "unsupported element type: ListItem: 1" in output
    assert "image file missing or unavailable: 1" in output


def test_query_passes_retrieved_image_and_table_to_generation(
    monkeypatch, capsys, valid_configuration
):
    context = [
        {
            "uuid": "image-id",
            "type": "Image",
            "content": "",
            "vision_summary": "Revenue rises.",
            "metadata": {
                "source": "report.pdf",
                "page": 3,
                "document_id": "report-1",
                "image_path": "chart.png",
            },
        },
        {
            "uuid": "table-id",
            "type": "Table",
            "content": "<table><tr><td>$10</td></tr></table>",
            "metadata": {
                "source": "report.pdf",
                "page": 2,
                "document_id": "report-1",
            },
        },
    ]
    embedder = MagicMock(name="embedder")
    generated = MagicMock(return_value="Revenue is $10.")
    retrieval = MagicMock(return_value=context)
    monkeypatch.setattr(main, "_check_query_collection", MagicMock())
    monkeypatch.setattr(main, "get_embedder", lambda: embedder)
    monkeypatch.setattr(main, "retrieve", retrieval)
    monkeypatch.setattr(main, "generate_answer", generated)
    monkeypatch.setattr(main.Path, "is_file", lambda self: str(self) == "chart.png")

    answer = main.query_documents(
        "What is the revenue?", document_id="report-1", k=2
    )

    assert answer == "Revenue is $10."
    retrieval.assert_called_once_with(
        "What is the revenue?", embedder, k=2, document_id="report-1"
    )
    generated.assert_called_once_with("What is the revenue?", context)
    output = capsys.readouterr().out
    assert "report.pdf, page 3" in output
    assert "report.pdf, page 2" in output
    assert "[document ID: report-1]" in output


def test_query_warns_on_missing_raw_element_and_invalid_image_path_with_summary(
    monkeypatch, capsys, valid_configuration
):
    image = {
        "uuid": "image-id",
        "type": "Image",
        "content": "",
        "vision_summary": "A chart summary.",
        "metadata": {"image_path": "missing-chart.png"},
        "original_element_missing": True,
    }
    monkeypatch.setattr(main, "_check_query_collection", lambda: None)
    monkeypatch.setattr(main, "get_embedder", lambda: MagicMock())
    monkeypatch.setattr(main, "retrieve", lambda *args, **kwargs: [image])
    generated = MagicMock(return_value="Answer")
    monkeypatch.setattr(main, "generate_answer", generated)
    monkeypatch.setattr(main.Path, "is_file", lambda self: False)

    assert main.query_documents("Summarize the chart.") == "Answer"

    generated.assert_called_once_with("Summarize the chart.", [image])
    error_output = capsys.readouterr().err
    assert "original Image 'image-id' is unavailable" in error_output
    assert "image 'image-id' is unavailable" in error_output


def test_query_fails_clearly_for_image_without_file_or_summary(
    monkeypatch, valid_configuration
):
    image = {
        "uuid": "image-id",
        "type": "Image",
        "content": "",
        "metadata": {"image_path": "missing-chart.png"},
    }
    monkeypatch.setattr(main, "_check_query_collection", lambda: None)
    monkeypatch.setattr(main, "get_embedder", lambda: MagicMock())
    monkeypatch.setattr(main, "retrieve", lambda *args, **kwargs: [image])
    monkeypatch.setattr(main.Path, "is_file", lambda self: False)
    generated = MagicMock()
    monkeypatch.setattr(main, "generate_answer", generated)

    with pytest.raises(main.PaperLensError, match="no available image file or text summary"):
        main.query_documents("Summarize the chart.")

    generated.assert_not_called()


def test_query_reports_generation_and_qdrant_errors(
    monkeypatch, valid_configuration
):
    monkeypatch.setattr(main, "_check_query_collection", lambda: None)
    monkeypatch.setattr(main, "get_embedder", lambda: MagicMock())
    monkeypatch.setattr(main, "retrieve", lambda *args, **kwargs: [{"type": "Title"}])
    monkeypatch.setattr(
        main, "generate_answer", MagicMock(side_effect=RuntimeError("API unavailable"))
    )
    with pytest.raises(main.PaperLensError, match="Answer generation failed"):
        main.query_documents("Question?")

    monkeypatch.setattr(
        main,
        "_check_query_collection",
        MagicMock(side_effect=main.PaperLensError("Qdrant unavailable")),
    )
    with pytest.raises(main.PaperLensError, match="Qdrant unavailable"):
        main.query_documents("Question?")


def test_retriever_filters_by_document_id_and_uses_matching_original(
    monkeypatch,
):
    from retrieval.multi_vector_retriever import retrieve

    indexed_doc = MagicMock()
    indexed_doc.page_content = "Summary"
    indexed_doc.metadata = {"uuid": "raw-id", "type": "Image"}
    raw_element = {"uuid": "raw-id", "type": "Image", "metadata": {}}
    store = MagicMock()
    store.similarity_search.return_value = [indexed_doc]
    get_store = MagicMock(return_value=store)
    load_raw = MagicMock(return_value=raw_element)
    monkeypatch.setattr("retrieval.vector_store.get_vector_store", get_store)
    monkeypatch.setattr("retrieval.multi_vector_retriever.load_raw_element", load_raw)
    embedder = MagicMock()

    results = retrieve("Question?", embedder, k=3, document_id="report-1")

    get_store.assert_called_once_with(embedder)
    store.similarity_search.assert_called_once()
    search_args = store.similarity_search.call_args
    assert search_args.args == ("Question?",)
    assert search_args.kwargs["k"] == 3
    condition = search_args.kwargs["filter"].must[0]
    assert condition.key == "metadata.document_id"
    assert condition.match.value == "report-1"
    load_raw.assert_called_once_with("raw-id")
    assert results == [raw_element]

    monkeypatch.setattr(
        "retrieval.multi_vector_retriever.load_raw_element",
        lambda doc_uuid: {**raw_element, "uuid": "wrong-id"},
    )
    with pytest.raises(ValueError, match="does not match indexed UUID"):
        retrieve("Question?", embedder, document_id="report-1")


def test_cli_reports_actionable_errors_without_traceback(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.test:6333")

    exit_code = main.main(["ingest", str(_write_pdf(tmp_path))])

    assert exit_code == 1
    error_output = capsys.readouterr().err
    assert "OPENAI_API_KEY is missing" in error_output
    assert "Traceback" not in error_output
