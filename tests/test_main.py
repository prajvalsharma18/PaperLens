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
    monkeypatch.setattr(main, "render_pdf_pages", lambda *args, **kwargs: [])
    monkeypatch.setattr(main, "save_document_manifest", lambda *args, **kwargs: None)


def test_cli_help_for_root_and_commands(capsys):
    for args in (
        ["--help"],
        ["ingest", "--help"],
        ["query", "--help"],
        ["index-visuals", "--help"],
        ["extract-data", "--help"],
    ):
        with pytest.raises(SystemExit) as exit_info:
            main.main(args)
        assert exit_info.value.code == 0
        assert "usage:" in capsys.readouterr().out


def test_structured_numeric_query_uses_local_evidence_without_generation(
    monkeypatch, capsys
):
    record = {
        "kind": "table",
        "status": "verified",
        "data": {
            "headers": ["Region", "Revenue"],
            "header_rows": [0],
            "rows": [["Region", "Revenue"], ["North", "9"], ["South", "12"]],
        },
        "document_id": "report-1",
        "source_element_id": "table-1",
        "source_file": "report.pdf",
        "page": 2,
        "extraction_method": "html_table_parser",
    }
    monkeypatch.setattr(
        main, "load_document_manifest", lambda _document_id: {"document_id": "report-1"}
    )
    monkeypatch.setattr(main, "load_structured_evidence", lambda _document_id: [record])
    validate = MagicMock(side_effect=AssertionError("configuration must not be required"))
    generate = MagicMock(side_effect=AssertionError("generation must not be called"))
    monkeypatch.setattr(main, "_validate_configuration", validate)
    monkeypatch.setattr(main, "generate_answer", generate)

    answer = main.query_documents(
        "Which region had the highest revenue chart?", document_id="report-1"
    )

    assert answer is not None
    assert "South: 12 Revenue" in answer
    validate.assert_not_called()
    generate.assert_not_called()
    assert "Answer:" in capsys.readouterr().out


def test_unsupported_numeric_visual_query_fails_closed_without_generation(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        main, "load_document_manifest", lambda _document_id: {"document_id": "report-1"}
    )
    monkeypatch.setattr(main, "load_structured_evidence", lambda _document_id: [])
    validate = MagicMock(side_effect=AssertionError("configuration must not be required"))
    generate = MagicMock(side_effect=AssertionError("generation must not be called"))
    monkeypatch.setattr(main, "_validate_configuration", validate)
    monkeypatch.setattr(main, "generate_answer", generate)

    answer = main.query_documents(
        "What is the exact value shown in the chart?", document_id="report-1"
    )

    assert answer is not None
    assert "does not support an unambiguous calculation" in answer
    validate.assert_not_called()
    generate.assert_not_called()
    assert "no values were inferred from appearance" in capsys.readouterr().out.lower()


def test_exact_page_plan_does_not_use_structured_answer_shortcut(monkeypatch):
    plan = main.create_query_plan("Which chart is shown on page 2?", "report-1")
    assert plan.retrieval_strategy == "exact_page"

    assert main._structured_numeric_answer("Which chart is shown on page 2?", plan) is None


def test_related_structured_evidence_is_attached_with_extraction_provenance(monkeypatch):
    record = {
        "record_id": "record-1",
        "kind": "chart",
        "document_id": "report-1",
        "source_element_id": "figure-1",
        "source_file": "report.pdf",
        "page": 3,
        "extraction_method": "axis_digitization",
        "status": "estimated",
        "data": {"values": [{"category": "North", "value": "12"}]},
        "warnings": ["estimated from chart pixels"],
        "limitations": [],
    }
    monkeypatch.setattr(main, "load_structured_evidence", lambda _doc: [record])
    plan = main.create_query_plan("Describe the chart on page 3", "report-1")
    items = main._related_structured_evidence(
        plan,
        [{
            "uuid": "figure-1",
            "type": "Image",
            "content": "",
            "metadata": {"document_id": "report-1", "page": 3},
        }],
    )
    assert len(items) == 1
    assert items[0]["metadata"]["record_id"] == "record-1"
    assert items[0]["metadata"]["quality_status"] == "estimated"
    assert items[0]["metadata"]["extraction_method"] == "axis_digitization"
    assert '"value": "12"' in items[0]["content"]


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


def test_query_cli_reports_invalid_page_range_without_service_access(capsys):
    assert main.main(["query", "Compare pages 4-2."]) == 1

    assert "Error: Invalid descending page range 4-2." in capsys.readouterr().err


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
    monkeypatch.setattr(
        main, "partition_pdf", lambda path, document_id=None: elements
    )
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
        main,         "partition_pdf", lambda path, document_id=None: [
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
        lambda path, document_id=None: [
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
    visual_search = MagicMock()
    monkeypatch.setattr(main, "search_visual_assets", visual_search)
    monkeypatch.setattr(main, "generate_answer", generated)
    monkeypatch.setattr(main.Path, "is_file", lambda self: str(self) == "chart.png")

    answer = main.query_documents(
        "What is the revenue?", document_id="report-1", k=2
    )

    assert answer == "Revenue is $10."
    retrieval.assert_called_once_with(
        "What is the revenue?", embedder, k=2, document_id="report-1"
    )
    generated.assert_called_once()
    assert generated.call_args.args[0] == "What is the revenue?"
    assert [item["metadata"]["evidence_id"] for item in generated.call_args.args[1]]
    assert [item["uuid"] for item in generated.call_args.args[1]] == [
        item["uuid"] for item in context
    ]
    visual_search.assert_not_called()
    output = capsys.readouterr().out
    assert "report.pdf, page 3" in output
    assert "report.pdf, page 2" in output
    assert "document ID: report-1" in output


def test_numeric_visual_query_without_structured_data_fails_closed(
    monkeypatch, capsys, valid_configuration
):
    chart = {
        "uuid": "chart-id",
        "type": "Image",
        "content": "",
        "vision_summary": "A chart with a tall red bar.",
        "metadata": {
            "document_id": "report",
            "page": 5,
            "image_path": "chart.png",
        },
    }
    monkeypatch.setattr(main, "_check_query_collection", lambda: None)
    monkeypatch.setattr(main, "get_embedder", lambda: "embedder")
    retrieval = MagicMock(return_value=[chart])
    monkeypatch.setattr(main, "retrieve", retrieval)
    monkeypatch.setattr(main, "search_visual_assets", lambda *args, **kwargs: [])
    monkeypatch.setattr(main.Path, "is_file", lambda self: str(self) == "chart.png")
    generated = MagicMock(return_value="The bar appears tallest.")
    monkeypatch.setattr(main, "generate_answer", generated)

    answer = main.query_documents("What is the exact value of the tallest bar?")

    assert answer is not None
    assert "cannot provide a numerical answer" in answer
    retrieval.assert_called_once_with(
        "What is the exact value of the tallest bar?",
        "embedder",
        k=4,
        document_id=None,
    )
    generated.assert_not_called()
    assert "no values were inferred from appearance" in capsys.readouterr().out.lower()


def test_index_visuals_cli_does_not_require_openai_credentials(
    monkeypatch, capsys, valid_configuration
):
    from retrieval.visual_indexer import VisualIndexResult

    monkeypatch.delenv("OPENAI_API_KEY")
    indexed = MagicMock(
        return_value=VisualIndexResult(
            parents_indexed=2,
            page_images_indexed=1,
            tiles_indexed=3,
            skipped_assets=0,
            skip_reasons={},
            model_name="openai/clip-vit-base-patch32",
            vector_dimension=512,
        )
    )
    monkeypatch.setattr(main, "index_document_visual_assets", indexed)

    assert main.main(
        ["index-visuals", "--document-id", "report", "--batch-size", "8"]
    ) == 0

    indexed.assert_called_once_with("report", batch_size=8)
    output = capsys.readouterr().out
    assert "Visual vector dimension: 512" in output
    assert "Visual child tiles upserted: 3" in output


def test_visual_query_fuses_text_and_visual_evidence_with_parent_context(
    tmp_path, monkeypatch, capsys, valid_configuration
):
    from PIL import Image

    crop_path = tmp_path / "chart-crop.png"
    store_path = tmp_path / "document-store"
    parent_path = store_path / "assets" / "report" / "chart-parent.png"
    Image.new("RGB", (12, 8), "red").save(crop_path)
    parent_path.parent.mkdir(parents=True)
    Image.new("RGB", (24, 16), "white").save(parent_path)
    monkeypatch.setenv("DOC_STORE_PATH", str(store_path))
    text_title = {
        "uuid": "title-id",
        "type": "Title",
        "content": "Quarterly revenue comparison",
        "metadata": {"document_id": "report", "page": 2},
    }
    text_parent = {
        "uuid": "figure-1",
        "type": "Image",
        "content": "",
        "vision_summary": "A quarterly revenue chart.",
        "metadata": {
            "document_id": "report",
            "page": 2,
            "image_path": str(parent_path),
        },
    }
    visual_tile = {
        "uuid": "tile-1",
        "type": "Image",
        "content": "",
        "metadata": {
            "document_id": "report",
            "page_number": 2,
            "image_path": str(crop_path),
            "parent_asset_path": "assets/report/chart-parent.png",
            "parent_id": "figure-1",
            "root_source_id": "figure-1",
            "chunk_type": "image_tile",
            "bbox": [0, 0, 12, 8],
            "bbox_coordinate_system": "parent_pixels",
        },
    }
    monkeypatch.setattr(main, "_check_query_collection", lambda: None)
    monkeypatch.setattr(main, "get_embedder", lambda: "text-embedder")
    monkeypatch.setattr(
        main, "retrieve", lambda *args, **kwargs: [text_title, text_parent]
    )
    visual_search = MagicMock(return_value=[visual_tile])
    monkeypatch.setattr(main, "search_visual_assets", visual_search)
    generated = MagicMock(return_value="The chart compares quarterly revenue.")
    monkeypatch.setattr(main, "generate_answer", generated)

    answer = main.query_documents(
        "What does the chart show?", document_id="report", k=2
    )

    assert answer == "The chart compares quarterly revenue."
    visual_search.assert_called_once_with(
        "What does the chart show?", limit=12, document_id="report"
    )
    context = generated.call_args.args[1]
    assert any(item["uuid"] == "title-id" for item in context)
    tile = next(item for item in context if item["uuid"] == "tile-1")
    assert tile["metadata"]["evidence_type"] == "visual_child_crop"
    assert "relative to its parent image" in tile["metadata"]["evidence_label"]
    assert sum(
        item.get("metadata", {}).get("image_path") in {
            str(parent_path),
            "assets/report/chart-parent.png",
        }
        for item in context
    ) == 1
    assert not any(
        item["metadata"].get("evidence_type") == "visual_parent_context"
        for item in context
    )
    assert capsys.readouterr().err == ""


def test_visual_query_keeps_semantic_match_when_visual_asset_disappears(
    tmp_path, monkeypatch, capsys, valid_configuration
):
    semantic = {
        "uuid": "shared-id",
        "type": "Title",
        "content": "A chart is introduced here.",
        "metadata": {"document_id": "report", "page": 1},
    }
    visual = {
        "uuid": "shared-id",
        "type": "Image",
        "content": "",
        "metadata": {
            "document_id": "report",
            "page": 1,
            "image_path": str(tmp_path / "removed-crop.png"),
            "chunk_type": "figure",
        },
    }
    monkeypatch.setattr(main, "_check_query_collection", lambda: None)
    monkeypatch.setattr(main, "get_embedder", lambda: "text-embedder")
    monkeypatch.setattr(main, "retrieve", lambda *args, **kwargs: [semantic])
    monkeypatch.setattr(main, "search_visual_assets", lambda *args, **kwargs: [visual])
    generated = MagicMock(return_value="The title introduces a chart.")
    monkeypatch.setattr(main, "generate_answer", generated)

    assert main.query_documents("What does the chart show?") == (
        "The title introduces a chart."
    )
    assert generated.call_args.args[1][0]["uuid"] == semantic["uuid"]
    assert generated.call_args.args[1][0]["content"] == semantic["content"]
    assert "Visual evidence assets are unavailable" in capsys.readouterr().err


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
    monkeypatch.setattr(main, "search_visual_assets", lambda *args, **kwargs: [])
    generated = MagicMock(return_value="Answer")
    monkeypatch.setattr(main, "generate_answer", generated)
    monkeypatch.setattr(main.Path, "is_file", lambda self: False)

    assert main.query_documents("Summarize the chart.") == "Answer"

    generation_context = generated.call_args.args[1]
    assert generation_context[0]["uuid"] == image["uuid"]
    assert generation_context[0]["metadata"]["evidence_id"].startswith("E-")
    assert any("whole-image evidence" in item["content"] for item in generation_context[1:])
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
    monkeypatch.setattr(main, "search_visual_assets", lambda *args, **kwargs: [])
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


def test_default_document_id_is_stable_per_path_and_content(tmp_path):
    from main import _default_document_id

    first_source = tmp_path / "first" / "report.pdf"
    second_source = tmp_path / "second" / "report.pdf"
    first_source.parent.mkdir()
    second_source.parent.mkdir()
    first_source.write_bytes(b"%PDF-1.7\nsame content")
    second_source.write_bytes(b"%PDF-1.7\nsame content")

    assert _default_document_id(first_source) == _default_document_id(first_source)
    assert _default_document_id(first_source) != _default_document_id(second_source)
    first_source.write_bytes(b"%PDF-1.7\nupdated content")
    assert _default_document_id(first_source) != _default_document_id(second_source)


def test_page_images_are_indexed_only_when_requested_and_not_duplicated(
    tmp_path, monkeypatch, valid_configuration
):
    from PIL import Image

    pdf_path = _write_pdf(tmp_path)
    extracted_image = tmp_path / "figure.png"
    Image.new("RGB", (8, 8), "blue").save(extracted_image)
    page_store = tmp_path / "doc_store" / "assets" / "pages"
    page_store.mkdir(parents=True)
    page_one = page_store / "page-00001.png"
    page_two = page_store / "page-00002.png"
    Image.new("RGB", (8, 8), "white").save(page_one)
    Image.new("RGB", (8, 8), "white").save(page_two)
    monkeypatch.setattr(main, "get_embedder", lambda: MagicMock())
    monkeypatch.setattr(main, "get_vector_store", lambda embedder: object())
    monkeypatch.setattr(
        main,
        "partition_pdf",
        lambda path, document_id=None: [
            {
                "uuid": "figure-id",
                "type": "Image",
                "content": "",
                "metadata": {
                    "page": 1,
                    "image_path": str(extracted_image),
                },
            },
            {
                "uuid": "text-id",
                "type": "Title",
                "content": "Scanned appendix",
                "metadata": {"page": 2},
            },
        ],
    )
    monkeypatch.setattr(
        main,
        "render_pdf_pages",
        lambda *args, **kwargs: [
            {
                "page_number": page_number,
                "image_path": f"assets/pages/page-{page_number:05}.png",
                "pixel_width": 8,
                "pixel_height": 8,
                "page_width": 8,
                "page_height": 8,
                "coordinate_system": {"origin": "top-left", "units": "pixels"},
                "rendering": {"library": "pypdfium2", "scale": 1},
            }
            for page_number in (1, 2)
        ],
    )
    monkeypatch.setattr(
        main,
        "summarise_images",
        lambda images: [{**image, "embed_content": "Figure summary"} for image in images],
    )
    saved = []
    indexed = {}
    monkeypatch.setattr(main, "save_raw_element", saved.append)
    monkeypatch.setattr(
        main,
        "upsert_documents",
        lambda documents, embedder: indexed.update(documents=documents),
    )

    assert main.ingest_pdf(str(pdf_path), include_page_images=True) == 3

    images = [item for item in indexed["documents"] if item["type"] == "Image"]
    assert [item["metadata"]["page"] for item in images] == [1, 2]
    page_fallback = next(item for item in images if item["metadata"]["page"] == 2)
    assert page_fallback["metadata"]["extraction_method"] == "pypdfium2_page_render"
    assert page_fallback["metadata"]["page_image_path"] == page_fallback["metadata"]["image_path"]
    assert all(item["metadata"]["document_id"] for item in saved)


def test_visual_chunking_is_opt_in_and_tiles_stay_out_of_the_vector_index(
    tmp_path, monkeypatch, capsys, valid_configuration
):
    from PIL import Image

    from retrieval.document_store import (
        load_document_manifest,
        save_document_manifest,
    )

    pdf_path = _write_pdf(tmp_path)
    image_path = tmp_path / "large figure.png"
    Image.new("RGB", (90, 60), "blue").save(image_path)
    elements = [
        {
            "uuid": "figure-id",
            "type": "Image",
            "content": "",
            "metadata": {"page": 1, "image_path": str(image_path)},
        },
        {
            "uuid": "text-id",
            "type": "NarrativeText",
            "content": "A report paragraph",
            "metadata": {"page": 1},
        },
    ]
    monkeypatch.setattr(main, "get_embedder", lambda: MagicMock())
    monkeypatch.setattr(main, "get_vector_store", lambda embedder: object())
    monkeypatch.setattr(main, "partition_pdf", lambda path, document_id=None: elements)
    monkeypatch.setattr(main, "render_pdf_pages", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        main,
        "summarise_images",
        lambda images: [
            {**image, "embed_content": "Figure summary"} for image in images
        ],
    )
    indexed = {}
    saved = []
    monkeypatch.setattr(main, "upsert_documents", lambda docs, _: indexed.update(docs=docs))
    monkeypatch.setattr(main, "save_raw_element", saved.append)
    monkeypatch.setattr(main, "save_document_manifest", save_document_manifest)
    monkeypatch.setattr(main, "load_document_manifest", load_document_manifest)
    save_document_manifest(
        "report-visuals", {"document_id": "report-visuals", "visual_chunks": [{"uuid": "old"}]}
    )
    chunker = MagicMock(wraps=main.chunk_image)
    monkeypatch.setattr(main, "chunk_image", chunker)

    assert main.ingest_pdf(
        str(pdf_path), document_id="report-visuals", tile_width=40,
        tile_height=30, tile_overlap=5, min_image_size=20,
    ) == 2
    chunker.assert_not_called()
    assert [doc["uuid"] for doc in indexed["docs"]] == ["figure-id", "text-id"]
    assert {doc["uuid"] for doc in saved} == {"figure-id", "text-id"}
    assert load_document_manifest("report-visuals")["visual_chunks"] == [{"uuid": "old"}]

    indexed.clear()
    saved.clear()
    assert main.ingest_pdf(
        str(pdf_path),
        document_id="report-visuals",
        chunk_visuals=True,
        tile_width=40,
        tile_height=30,
        tile_overlap=5,
        min_image_size=20,
    ) == 2
    assert chunker.call_count == 1
    manifest_chunks = load_document_manifest("report-visuals")["visual_chunks"]
    assert len(manifest_chunks) == 9
    assert all(record["type"] == "ImageTile" for record in manifest_chunks)
    assert [doc["uuid"] for doc in indexed["docs"]] == ["figure-id", "text-id"]
    assert {doc["uuid"] for doc in saved} == {"figure-id", "text-id"}
    assert not {doc["uuid"] for doc in indexed["docs"]}.intersection(
        record["uuid"] for record in manifest_chunks
    )
    report = capsys.readouterr().out
    assert "Visual child chunks generated: 9" in report
    assert "Extracted-figure parents chunked: 1" in report


@pytest.mark.parametrize(
    ("include_page_images", "expected_parents"),
    [(False, 0), (True, 1)],
)
def test_page_renders_are_chunked_only_when_page_image_indexing_is_enabled(
    tmp_path, monkeypatch, capsys, valid_configuration, include_page_images, expected_parents
):
    from PIL import Image

    pdf_path = _write_pdf(tmp_path)
    page_path = tmp_path / "doc_store" / "assets" / "report" / "pages" / "page.png"
    page_path.parent.mkdir(parents=True)
    Image.new("RGB", (60, 45), "white").save(page_path)
    monkeypatch.setattr(main, "get_embedder", lambda: MagicMock())
    monkeypatch.setattr(main, "get_vector_store", lambda embedder: object())
    monkeypatch.setattr(
        main,
        "partition_pdf",
        lambda path, document_id=None: [
            {
                "uuid": "page-title",
                "type": "Title",
                "content": "Page title",
                "metadata": {"page": 1},
            }
        ],
    )
    monkeypatch.setattr(
        main,
        "render_pdf_pages",
        lambda *args, **kwargs: [
            {
                "page_number": 1,
                "image_path": "assets/report/pages/page.png",
                "pixel_width": 60,
                "pixel_height": 45,
                "page_width": 60,
                "page_height": 45,
                "coordinate_system": {"origin": "top-left", "units": "pixels"},
            }
        ],
    )
    monkeypatch.setattr(
        main,
        "summarise_images",
        lambda images: [
            {**image, "embed_content": "Page image summary"} for image in images
        ],
    )
    indexed = {}
    monkeypatch.setattr(
        main, "upsert_documents", lambda docs, _: indexed.update(docs=docs)
    )
    monkeypatch.setattr(main, "save_raw_element", lambda _: None)
    monkeypatch.setattr(main, "save_document_manifest", lambda *args, **kwargs: None)

    assert main.ingest_pdf(
        str(pdf_path),
        document_id="report",
        include_page_images=include_page_images,
        chunk_visuals=True,
        tile_width=20,
        tile_height=20,
        tile_overlap=5,
        min_image_size=20,
    ) == 1 + include_page_images
    assert "Visual parents discovered: {}".format(expected_parents) in capsys.readouterr().out
    if include_page_images:
        assert len(indexed["docs"]) == 2
        page = next(
            doc for doc in indexed["docs"] if doc["type"] == "Image"
        )
        assert page["metadata"]["extraction_method"] == "pypdfium2_page_render"
    else:
        assert [doc["uuid"] for doc in indexed["docs"]] == ["page-title"]


def test_page_question_fetches_manifest_render_and_page_elements_exactly(
    tmp_path, monkeypatch, capsys, valid_configuration
):
    from PIL import Image

    paths = {}
    for page_number in (1, 2, 3):
        page_path = tmp_path / f"page-{page_number}.png"
        Image.new("RGB", (12, 8), (page_number * 40, 0, 0)).save(page_path)
        paths[page_number] = str(page_path)
    manifest = {
        "document_id": "report-one",
        "source_file": "report.pdf",
        "source_path": r"C:\papers\report.pdf",
        "page_count": 3,
        "pages": [
            {"page_number": page_number, "image_path": paths[page_number]}
            for page_number in (1, 2, 3)
        ],
    }
    elements_read = []

    def page_elements(document_id, page_number):
        elements_read.append((document_id, page_number))
        return [
            {
                "uuid": f"title-{page_number}",
                "type": "Title",
                "content": f"Page {page_number} title",
                "metadata": {
                    "document_id": document_id,
                    "page": page_number,
                    "source_file": "report.pdf",
                },
            }
        ]

    monkeypatch.setattr(main, "load_document_manifest", lambda _: manifest)
    monkeypatch.setattr(main, "load_page_elements", page_elements)
    monkeypatch.setattr(main, "_check_query_collection", MagicMock())
    monkeypatch.setattr(main, "get_embedder", MagicMock())
    monkeypatch.setattr(main, "retrieve", MagicMock())
    monkeypatch.setattr(main, "search_visual_assets", MagicMock())
    generated = MagicMock(return_value="Page-specific answer")
    monkeypatch.setattr(main, "generate_answer", generated)

    answer = main.query_documents(
        "What is shown on page 2?", document_id="report-one"
    )

    assert answer == "Page-specific answer"
    context = generated.call_args.args[1]
    assert [item["metadata"]["page"] for item in context] == [2, 2]
    assert context[0]["metadata"]["image_path"] == paths[2]
    assert context[0]["metadata"]["require_image_evidence"]
    assert context[1]["content"] == "Page 2 title"
    assert context[0]["metadata"]["evidence_label"] == (
        "Document report-one, page 2 (full-page render)"
    )
    assert elements_read == [("report-one", 2)]
    main._check_query_collection.assert_not_called()
    main.get_embedder.assert_not_called()
    main.retrieve.assert_not_called()
    main.search_visual_assets.assert_not_called()
    output = capsys.readouterr().out
    assert "report.pdf, page 2" in output
    assert "page 1" not in output


def test_page_comparison_preserves_requested_order_and_includes_every_image(
    tmp_path, monkeypatch, valid_configuration
):
    from PIL import Image

    paths = {}
    for page_number in (1, 2, 3):
        page_path = tmp_path / f"comparison-page-{page_number}.png"
        Image.new("RGB", (10, 8), "white").save(page_path)
        paths[page_number] = str(page_path)
    manifest = {
        "document_id": "report-two",
        "source_file": "layouts.pdf",
        "page_count": 3,
        "pages": [
            {"page_number": page_number, "image_path": paths[page_number]}
            for page_number in (1, 2, 3)
        ],
    }
    read_pages = []
    monkeypatch.setattr(main, "load_document_manifest", lambda _: manifest)
    monkeypatch.setattr(
        main,
        "load_page_elements",
        lambda document_id, page_number: (
            read_pages.append((document_id, page_number)) or []
        ),
    )
    monkeypatch.setattr(main, "_check_query_collection", MagicMock())
    generated = MagicMock(return_value="Comparison")
    monkeypatch.setattr(main, "generate_answer", generated)

    main.query_documents(
        "Compare layouts on pages 3, 1, and 2.", document_id="report-two"
    )

    context = generated.call_args.args[1]
    page_images = [
        item
        for item in context
        if item.get("metadata", {}).get("evidence_type") == "full_page_render"
    ]
    assert [item["metadata"]["page"] for item in page_images] == [3, 1, 2]
    assert [item["metadata"]["image_path"] for item in page_images] == [
        paths[3],
        paths[1],
        paths[2],
    ]
    assert [item["metadata"]["evidence_label"] for item in page_images] == [
        f"Document report-two, page {page} (full-page render)"
        for page in (3, 1, 2)
    ]
    assert read_pages == [("report-two", 3), ("report-two", 1), ("report-two", 2)]
    assert any("Child tiles are not searchable" in item["content"] for item in context)
    main._check_query_collection.assert_not_called()


def test_page_question_requires_unambiguous_document_selection(
    monkeypatch, valid_configuration
):
    monkeypatch.setattr(
        main,
        "list_document_manifests",
        lambda: [
            {"document_id": "report-one", "pages": []},
            {"document_id": "report-two", "pages": []},
        ],
    )
    generated = MagicMock()
    monkeypatch.setattr(main, "generate_answer", generated)

    with pytest.raises(main.PaperLensError, match="multiple documents"):
        main.query_documents("What is shown on page 1?")

    generated.assert_not_called()


def test_out_of_range_page_fails_without_semantic_substitution(
    tmp_path, monkeypatch, valid_configuration
):
    from PIL import Image

    page_path = tmp_path / "only-page.png"
    Image.new("RGB", (10, 10), "white").save(page_path)
    manifest = {
        "document_id": "three-page-report",
        "source_file": "three-pages.pdf",
        "page_count": 3,
        "pages": [{"page_number": 1, "image_path": str(page_path)}],
    }
    monkeypatch.setattr(main, "load_document_manifest", lambda _: manifest)
    monkeypatch.setattr(main, "_check_query_collection", MagicMock())
    generated = MagicMock()
    monkeypatch.setattr(main, "generate_answer", generated)

    with pytest.raises(main.PaperLensError, match="Page 4 is unavailable.*3 page"):
        main.query_documents(
            "Compare page 1 and page 4.", document_id="three-page-report"
        )

    main._check_query_collection.assert_not_called()
    generated.assert_not_called()


def test_missing_render_fails_clearly_and_does_not_call_generation(
    monkeypatch, valid_configuration
):
    manifest = {
        "document_id": "no-render",
        "source_file": "unrendered.pdf",
        "page_count": 1,
        "pages": [{"page_number": 1}],
    }
    monkeypatch.setattr(main, "load_document_manifest", lambda _: manifest)
    generated = MagicMock()
    monkeypatch.setattr(main, "generate_answer", generated)

    with pytest.raises(main.PaperLensError, match="no rendered image"):
        main.query_documents("What is shown on page 1?", document_id="no-render")

    generated.assert_not_called()


def test_unselected_page_query_uses_only_the_single_available_manifest(
    tmp_path, monkeypatch, valid_configuration
):
    from PIL import Image

    image_path = tmp_path / "selected-page.png"
    Image.new("RGB", (8, 8), "white").save(image_path)
    manifest = {
        "document_id": "only-report",
        "source_file": "only.pdf",
        "page_count": 1,
        "pages": [{"page_number": 1, "image_path": str(image_path)}],
    }
    monkeypatch.setattr(main, "list_document_manifests", lambda: [manifest])
    monkeypatch.setattr(
        main, "load_page_elements", lambda document_id, page_number: []
    )
    generated = MagicMock(return_value="Answer")
    monkeypatch.setattr(main, "generate_answer", generated)
    monkeypatch.setattr(main, "_check_query_collection", MagicMock())

    assert main.query_documents("What is on page 1?") == "Answer"
    assert generated.call_args.args[1][0]["metadata"]["document_id"] == "only-report"
    main._check_query_collection.assert_not_called()


def test_optional_ocr_indexes_text_for_pages_without_parser_text(
    tmp_path, monkeypatch, valid_configuration
):
    pdf_path = _write_pdf(tmp_path)
    monkeypatch.setattr(main, "get_embedder", lambda: MagicMock())
    monkeypatch.setattr(main, "get_vector_store", lambda embedder: object())
    monkeypatch.setattr(
        main, "partition_pdf", lambda path, document_id=None: []
    )
    monkeypatch.setattr(
        main,
        "render_pdf_pages",
        lambda *args, **kwargs: [
            {
                "page_number": 1,
                "image_path": "assets/pages/page-00001.png",
                "pixel_width": 100,
                "pixel_height": 50,
                "page_width": 72,
                "page_height": 36,
                "coordinate_system": {"origin": "top-left", "units": "pixels"},
                "rendering": {"library": "pypdfium2", "scale": 1},
            }
        ],
    )
    ocr_result = {
        "text": "Scanned report title",
        "bbox": [4, 5, 50, 15],
        "page_width": 100,
        "page_height": 50,
        "coordinate_system": {"origin": "top-left", "units": "pixels"},
        "extraction_method": "tesseract_ocr",
    }
    ocr = MagicMock(return_value=ocr_result)
    monkeypatch.setattr(main, "ocr_image", ocr)
    saved = []
    indexed = {}
    monkeypatch.setattr(main, "save_raw_element", saved.append)
    monkeypatch.setattr(
        main,
        "upsert_documents",
        lambda documents, embedder: indexed.update(documents=documents),
    )

    assert main.ingest_pdf(str(pdf_path), ocr=True, ocr_language="eng") == 1

    ocr.assert_called_once()
    element = indexed["documents"][0]
    assert element["content"] == "Scanned report title"
    assert element["metadata"]["page"] == 1
    assert element["metadata"]["bbox"] == [4, 5, 50, 15]
    assert element["metadata"]["extraction_method"] == "tesseract_ocr"
    assert element["metadata"]["page_image_path"] == "assets/pages/page-00001.png"
