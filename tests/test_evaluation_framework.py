import json

import pytest

from evaluation.answer_metrics import evaluate_answer_case
from evaluation.benchmark import (
    _generation_input_experiment,
    _relevant_ranked_ids,
    _retrieval_experiment,
    _write_report,
    build_report,
    main as benchmark_main,
)
from evaluation.dataset import DatasetValidationError, load_dataset, validate_dataset
from evaluation.structured_extraction_benchmark import run_benchmark


def _dataset():
    return {
        "schema_version": 1,
        "dataset_id": "controlled",
        "cases": [{
            "case_id": "one", "split": "test", "category": "region",
            "query": "What is in this region?",
            "source": {"kind": "synthetic_fixture", "annotation_status": "controlled"},
            "retrieval_evaluation": True,
            "relevant_evidence": [{
                "id": "tile-2", "granularity": "region", "relevance": 2,
                "annotation": "Exact requested region.",
            }],
        }],
    }


def test_dataset_validation_rejects_duplicate_case_ids():
    value = _dataset()
    value["cases"].append(dict(value["cases"][0]))
    with pytest.raises(DatasetValidationError, match="Duplicate case_id"):
        validate_dataset(value)


def test_dataset_loader_reads_versioned_schema(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(_dataset()), encoding="utf-8")
    assert load_dataset(path)["dataset_id"] == "controlled"


def test_real_retrieval_cases_require_a_document_id():
    value = _dataset()
    case = value["cases"][0]
    case["source"] = {"kind": "real_document", "annotation_status": "manual"}
    with pytest.raises(DatasetValidationError, match="document_id"):
        validate_dataset(value)


def test_retrieval_matching_keeps_ranks_and_does_not_promote_parent_to_region():
    case = _dataset()["cases"][0]
    candidates = [
        {"uuid": "unrelated", "metadata": {"parent_id": "tile-2"}},
        {"uuid": "tile-2", "metadata": {"parent_id": "figure-1"}},
    ]
    assert _relevant_ranked_ids(case, candidates) == ["unrelated", "tile-2"]


def test_retrieval_failures_remain_in_metric_denominator(monkeypatch):
    import retrieval.vector_store

    monkeypatch.setattr(
        retrieval.vector_store, "_get_client",
        lambda: (_ for _ in ()).throw(RuntimeError("Qdrant offline")),
    )
    case = _dataset()["cases"][0]
    case["query"] = "What does the figure show?"
    case["source"] = {
        "kind": "real_document", "document_id": "doc-1",
        "annotation_status": "manually-labelled",
    }
    result = _retrieval_experiment(
        {"cases": [case]}, split=None, allow_remote=False, k_values=(1, 3)
    )
    assert len(result["per_case"]) == 3
    assert all(row["status"] == "failed" for row in result["per_case"])
    assert all(row["failure_reason"] for row in result["per_case"])
    assert all(row["case_count"] == 1 and row["failed"] == 1 for row in result["aggregates"])
    assert all(row["metrics"]["recall@1"] == 0 for row in result["aggregates"])


def test_page_comparison_cases_are_reported_as_controls_without_qdrant(monkeypatch):
    import retrieval.vector_store

    monkeypatch.setattr(
        retrieval.vector_store, "_get_client",
        lambda: (_ for _ in ()).throw(AssertionError("similarity index must not be used")),
    )
    case = _dataset()["cases"][0]
    case["query"] = "Compare the layouts on pages 2 and 4"
    result = _retrieval_experiment(
        {"cases": [case]}, split=None, allow_remote=False, k_values=(1, 3)
    )
    assert result["eligible_similarity_cases"] == 0
    assert result["page_specific_controls"][0]["status"] == "excluded_similarity_bypass"
    assert result["per_case"] == []


def test_page_labels_match_page_metadata_but_region_labels_remain_exact():
    case = _dataset()["cases"][0]
    case["relevant_evidence"] = [{
        "id": "page:doc-1:3", "granularity": "page", "relevance": 1,
        "annotation": "Supporting page.",
    }]
    candidate = {"uuid": "figure-1", "metadata": {"document_id": "doc-1", "page": 3}}
    assert _relevant_ranked_ids(case, [candidate]) == ["page:doc-1:3"]


def test_answer_checks_facts_citations_units_forbidden_claims_and_abstention():
    expected = {
        "required_facts": ["12.5"], "forbidden_facts": ["exactly measured"],
        "required_citations": ["[E-abc]"],
        "available_evidence_ids": ["E-abc"],
        "numeric": {"value": "12.5", "unit": "million", "tolerance": "0"},
        "should_abstain": False,
    }
    assert evaluate_answer_case("12.5 million [E-abc]", expected)["passed"]
    assert not evaluate_answer_case("12.5 [E-fffffffff0] exactly measured", expected)["passed"]
    assert evaluate_answer_case(
        "I cannot determine this from available evidence.",
        {"required_facts": [], "forbidden_facts": [], "required_citations": [], "should_abstain": True},
    )["passed"]


def test_offline_report_distinguishes_synthetic_extraction_and_mocked_answers():
    dataset = load_dataset("evaluation/datasets/answer-fixtures.json")
    report = build_report("all", dataset)
    assert report["retrieval"]["status"] == "pending_no_labelled_retrieval_dataset"
    assert report["extraction"]["evaluation_kind"] == "synthetic_fixture_execution"
    assert report["extraction"]["fixture_case_count"] == 4
    assert report["extraction"]["table_cells_correct"] == 6
    assert report["extraction"]["table_cells_denominator"] > 0
    assert report["extraction"]["chart_category_associations_correct"] == 2
    assert report["extraction"]["chart_category_associations_denominator"] == 2
    assert report["extraction"]["estimated_value_pixel_mae"] < 0.1
    assert report["extraction"]["estimated_value_expected_examples"] == 2
    assert report["extraction"]["direct_label_examples"] == 0
    assert report["extraction"]["provenance_denominator"] == 4
    assert report["extraction"]["status_correct"] == 4
    assert report["extraction"]["incomplete_records"] == 1
    assert report["extraction"]["unsupported_records"] == 1
    assert report["answers"]["evaluation_kind"] == "deterministic_controlled_fixture_checks"
    assert report["answers"]["passed"] == 2
    assert all(
        item["execution_kind"] == "mocked_result"
        for item in report["answers"]["per_case"]
    )


def test_report_writes_json_and_markdown_without_overwriting(tmp_path):
    report = {
        "run_id": "fixed", "timestamp_utc": "now", "experiment": "extraction",
        "execution_kind": "synthetic_fixture_execution", "case_count": 2,
        "results": {}, "limitations": [],
    }
    output = tmp_path / "report.json"
    paths = _write_report(report, output)
    assert all(path.is_file() for path in paths)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _write_report(report, output)


def test_offline_benchmark_command_writes_a_structured_report(tmp_path, capsys):
    dataset = str(
        __import__("pathlib").Path(__file__).parents[1]
        / "evaluation" / "datasets" / "answer-fixtures.json"
    )
    output = tmp_path / "offline-report.json"
    result = benchmark_main([
        "--mode", "all", "--dataset", dataset, "--output", str(output),
    ])
    assert result == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["retrieval"]["status"] == "pending_no_labelled_retrieval_dataset"
    assert report["extraction"]["evaluation_kind"] == "synthetic_fixture_execution"
    assert report["answers"]["passed"] == 2
    assert "Markdown summary:" in capsys.readouterr().out


def test_benchmark_command_help_is_available(capsys):
    with pytest.raises(SystemExit) as exit_info:
        benchmark_main(["--help"])
    assert exit_info.value.code == 0
    assert "--mode" in capsys.readouterr().out


def test_generation_input_experiment_holds_text_constant_and_marks_mocks(monkeypatch):
    import main
    import retrieval.embedder

    monkeypatch.setattr(retrieval.embedder, "get_embedder", lambda: "fake-embedder")
    textual = {
        "uuid": "text-1", "type": "Title", "content": "The figure compares sales.",
        "metadata": {"document_id": "doc-1", "page": 1},
    }
    visual = {
        "uuid": "figure-1", "type": "Image", "content": "",
        "metadata": {"document_id": "doc-1", "root_source_id": "figure-1"},
    }
    image = {
        "uuid": "figure-1", "type": "Image", "content": "",
        "metadata": {"evidence_label": "Document doc-1, page 1"},
    }
    monkeypatch.setattr(main, "_visual_evidence_items", lambda _visual, _text: [image])
    calls = []

    def generate(_query, context):
        calls.append(context)
        return "The figure compares sales."

    case = {
        "case_id": "generation-control", "split": "test",
            "query": "Describe the figure.",
            "source": {
                "kind": "real_document", "document_id": "doc-1",
                "annotation_status": "fixture test stub",
            },
        "generation_evaluation": {"expected": {"required_facts": ["compares sales"]}},
    }
    result = _generation_input_experiment(
        {"cases": [case]}, split=None, allow_remote=True,
        retrieval_fn=lambda *_args, **_kwargs: [textual],
        visual_fn=lambda *_args, **_kwargs: [visual],
        generate_fn=generate,
    )
    assert result["evaluation_kind"] == "mocked_pipeline_results"
    assert result["generation_calls"] == 2
    assert calls[0] == calls[1][:1]
    assert calls[0][0]["uuid"] == "text-1"
    assert calls[1][1] == image
    assert result["per_case"][0]["text_only"]["passed"]
    assert result["per_case"][0]["text_plus_images"]["passed"]


def test_generation_input_cli_requires_explicit_generation_opt_in(tmp_path):
    result = benchmark_main([
        "--mode", "generation-input", "--output", str(tmp_path / "blocked.json"),
    ])
    assert result == 2
    assert not (tmp_path / "blocked.json").exists()


def test_synthetic_extraction_metrics_report_denominators_and_tolerance():
    result = run_benchmark()
    assert result["table_cells_denominator"] == 6
    assert result["chart_category_associations_denominator"] == 2
    assert result["estimated_value_examples"] == 2
    assert result["estimated_value_expected_examples"] == 2
    assert result["estimated_value_mae_tolerance"] == 0.1
    assert result["direct_label_examples"] == 0
    assert result["unsupported_records"] == 1
