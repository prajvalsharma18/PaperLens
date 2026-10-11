"""Reproducible, offline-safe PaperLens retrieval/extraction/answer evaluation."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.answer_metrics import evaluate_answer_case
from evaluation.dataset import DatasetValidationError, load_dataset


DEFAULT_K = (1, 3, 5)


def _ids(candidate: dict[str, Any]) -> set[str]:
    metadata = candidate.get("metadata", {})
    primary = candidate.get("uuid") or metadata.get("chunk_id") or metadata.get("element_id")
    return {str(primary)} if primary else set()


def _case_relevance(case: dict[str, Any]) -> dict[str, int]:
    return {
        str(item["id"]): int(item.get("relevance", 1))
        for item in case.get("relevant_evidence", [])
        if int(item.get("relevance", 1)) > 0
    }


def _relevant_ranked_ids(
    case: dict[str, Any], candidates: list[dict[str, Any]]
) -> list[str]:
    labels = case.get("relevant_evidence", [])
    ranked: list[str] = []
    for candidate in candidates:
        identifiers = _ids(candidate)
        metadata = candidate.get("metadata", {})
        page = metadata.get("page", metadata.get("page_number"))
        document_id = metadata.get("document_id") or case.get("source", {}).get("document_id")
        matched: list[str] = []
        for label in labels:
            expected_id = str(label["id"])
            if expected_id in identifiers:
                matched.append(expected_id)
            if label.get("granularity") == "page" and page is not None and document_id:
                page_ids = {
                    f"page:{document_id}:{page}",
                    f"{document_id}:page:{page}",
                    str(page),
                }
                if expected_id in page_ids:
                    matched.append(expected_id)
        # Keep every non-relevant result in its true rank position. A candidate
        # can satisfy one labelled identity at its rank; it is not duplicated.
        ranked.append(matched[0] if matched else next(iter(identifiers), ""))
    return ranked


def _retrieval_markdown(retrieval: dict[str, Any]) -> list[str]:
    lines = [
        "## Retrieval results", "",
        "Empty relevance sets score zero and remain in the denominator; failed retrievals are also retained as empty rankings. Explicit page questions are listed separately and excluded from similarity-method scores.",
        "",
        "| Method | Category | Cases | Valid labels | No results | Failed | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    aggregates = retrieval.get("aggregates", [])
    for row in aggregates:
        metrics = row.get("metrics", {})
        lines.append(
            f"| {row['method']} | {row['category']} | {row['case_count']} | {row['valid_ground_truth']} | {row['no_results']} | {row['failed']} | "
            + " | ".join(f"{metrics.get(key, 0):.3f}" for key in ("recall@1", "recall@3", "recall@5", "mrr", "ndcg@5"))
            + " |"
        )
    if not aggregates:
        status = retrieval.get("status", "no eligible labelled cases")
        lines.append(f"| pending | {status} | 0 | 0 | 0 | 0 | — | — | — | — | — |")
    lines.extend([
        "",
        f"Page-specific controls excluded from similarity metrics: {len(retrieval.get('page_specific_controls', []))}.",
        f"Cases without retrieval labels excluded: {len(retrieval.get('unlabelled_controls', []))}.",
        "",
    ])
    return lines


def _write_report(report: dict[str, Any], output: str | Path) -> tuple[Path, Path]:
    json_path = Path(output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path = json_path.with_suffix(".md")
    if json_path.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation report: {json_path}")
    if markdown_path.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation summary: {markdown_path}")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        f"# PaperLens evaluation report: {report['run_id']}",
        "",
        f"- Experiment: `{report['experiment']}`",
        f"- Execution: `{report['execution_kind']}`",
        f"- Cases selected: {report.get('case_count', 0)}",
        f"- Timestamp (UTC): {report['timestamp_utc']}",
        "",
    ]
    if report.get("experiment") in {"retrieval", "retrieval_comparison"}:
        lines.extend(_retrieval_markdown(report.get("retrieval", report)))
    else:
        if report.get("retrieval"):
            lines.extend(_retrieval_markdown(report["retrieval"]))
        elif report["experiment"] == "all":
            lines.extend(_retrieval_markdown({"status": "no labelled corpus supplied"}))
        extraction = report.get("extraction")
        if extraction:
            lines.extend([
                "## Synthetic extraction results", "",
                f"- Table cell exact match: {extraction['table_cells_correct']}/{extraction['table_cells_denominator']} ({extraction['table_cell_exact_match']:.3f})",
                f"- Table header exact match: {extraction['table_header_rows_correct']}/{extraction['table_header_rows_denominator']}",
                f"- Table row structure exact match: {extraction['table_row_structure_correct']}/{extraction['table_row_structure_denominator']}",
                f"- Provenance correct: {extraction['provenance_correct']}/{extraction['provenance_denominator']}",
                f"- Chart category associations: {extraction['chart_category_associations_correct']}/{extraction['chart_category_associations_denominator']}",
                f"- Estimated-value MAE: {extraction['estimated_value_pixel_mae']} over n={extraction['estimated_value_examples']}/{extraction['estimated_value_expected_examples']}; missing={extraction['estimated_value_missing_examples']}; tolerance={extraction['estimated_value_mae_tolerance']}",
                f"- Incomplete records: {extraction['incomplete_records']}/{extraction['status_record_denominator']}; unsupported: {extraction['unsupported_records']}/{extraction['status_record_denominator']}",
                f"- Extraction status correct: {extraction['status_correct']}/{extraction['status_record_denominator']}",
                f"- Direct OCR labels: not measured (n={extraction['direct_label_examples']})", "",
            ])
        answers = report.get("answers")
        if answers:
            lines.extend([
                "## Controlled answer checks", "",
                f"- Passed fixture checks: {answers['passed']}/{answers['denominator']}", "",
            ])
        generation = report.get("generation_input")
        if generation:
            lines.extend([
                "## Generation-input comparison", "",
                f"- Execution: {generation['evaluation_kind']}",
                f"- Cases selected: {generation['case_count']}",
                f"- Generation calls attempted: {generation['generation_calls']}",
                f"- Cost: {generation['cost']}",
                f"- Method: {generation['note']}", "",
            ])
    lines.extend(["", "## Limitations", "", *[f"- {item}" for item in report.get("limitations", [])], ""])
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path


def _retrieval_experiment(
    dataset: dict[str, Any], *, split: str | None, allow_remote: bool, k_values: tuple[int, ...]
) -> dict[str, Any]:
    from evaluation.retrieval_metrics import evaluate_retrieval
    from retrieval.hybrid_retriever import select_hybrid_candidates
    from retrieval.query_router import create_query_plan

    selected = [case for case in dataset["cases"] if split is None or case["split"] == split]
    rows: list[dict[str, Any]] = []
    methods = ("text", "visual", "hybrid")
    eligible: list[dict[str, Any]] = []
    page_controls = []
    unlabelled_controls = []
    non_real_controls = []
    for case in selected:
        if not case.get("retrieval_evaluation"):
            unlabelled_controls.append({"case_id": case["case_id"], "status": "excluded_no_retrieval_judgements", "category": case["category"]})
            continue
        plan = create_query_plan(case["query"], case.get("source", {}).get("document_id"))
        if plan.retrieval_strategy == "exact_page" or case.get("query_intent") in {"page_question", "page_comparison"}:
            page_controls.append({"case_id": case["case_id"], "status": "excluded_similarity_bypass", "category": case["category"]})
        elif case.get("source", {}).get("kind") != "real_document":
            non_real_controls.append({"case_id": case["case_id"], "status": "excluded_non_real_corpus_case", "source_kind": case.get("source", {}).get("kind")})
        else:
            eligible.append(case)

    text_count: int | None = None
    visual_count: int | None = None
    index_errors: dict[str, str] = {}
    if eligible:
        try:
            from retrieval.vector_store import _get_client

            client = _get_client()
            names = {item.name for item in client.get_collections().collections}
            text_collection = os.getenv("COLLECTION_NAME", "rag_documents")
            visual_collection = os.getenv("VISUAL_COLLECTION_NAME", "rag_visuals")
            if text_collection not in names:
                text_count = 0
            else:
                text_count = int(client.count(collection_name=text_collection).count)
            if visual_collection not in names:
                visual_count = 0
            else:
                visual_count = int(client.count(collection_name=visual_collection).count)
        except Exception as exc:
            index_errors = {
                "text": f"Could not inspect Qdrant indexes: {type(exc).__name__}: {exc}",
                "visual": f"Could not inspect Qdrant indexes: {type(exc).__name__}: {exc}",
            }
    embedder = None
    embedder_error = None
    if eligible and not index_errors and text_count and allow_remote:
        try:
            from retrieval.embedder import get_embedder

            embedder = get_embedder()
        except Exception as exc:
            embedder_error = f"Text embedding initialization failed: {type(exc).__name__}: {exc}"
    elif eligible and text_count:
        embedder_error = "Remote text embeddings require --allow-remote-text-embeddings."

    for case in eligible:
        doc_id = case.get("source", {}).get("document_id")
        results: dict[str, list[dict[str, Any]]] = {}
        failures: dict[str, str] = {}
        if "text" in index_errors:
            failures["text"] = index_errors["text"]
        elif not text_count:
            results["text"] = []
        elif embedder is None:
            failures["text"] = embedder_error or "Text embedding model unavailable."
        else:
            try:
                from retrieval.multi_vector_retriever import retrieve

                results["text"] = retrieve(case["query"], embedder, k=max(k_values), document_id=doc_id)
            except Exception as exc:
                failures["text"] = f"{type(exc).__name__}: {exc}"
        if "visual" in index_errors:
            failures["visual"] = index_errors["visual"]
        elif not visual_count:
            results["visual"] = []
        else:
            try:
                from retrieval.visual_indexer import search_visual_assets

                # Prevent Transformers from downloading missing weights during a benchmark.
                old_offline = os.environ.get("HF_HUB_OFFLINE")
                os.environ["HF_HUB_OFFLINE"] = "1"
                try:
                    results["visual"] = search_visual_assets(
                        case["query"], limit=max(k_values), document_id=doc_id,
                        client=client,
                    )
                finally:
                    if old_offline is None:
                        os.environ.pop("HF_HUB_OFFLINE", None)
                    else:
                        os.environ["HF_HUB_OFFLINE"] = old_offline
            except Exception as exc:
                failures["visual"] = f"{type(exc).__name__}: {exc}"
        if "text" in results and "visual" in results:
            try:
                results["hybrid"] = select_hybrid_candidates(
                    results["text"], results["visual"],
                    text_limit=max(k_values), visual_limit=max(k_values),
                )
            except Exception as exc:
                failures["hybrid"] = f"{type(exc).__name__}: {exc}"
        else:
            failures["hybrid"] = "Hybrid requires successful text and visual retrieval."

        relevance_map = _case_relevance(case)
        for method in methods:
            candidates = results.get(method)
            if candidates is None:
                rows.append({"case_id": case["case_id"], "category": case["category"], "method": method, "status": "failed", "failure_reason": failures.get(method, "No result returned."), "ranking": [], "relevant_ranked_ids": [], "relevance": relevance_map})
            else:
                ranked = [next(iter(_ids(candidate)), "") for candidate in candidates]
                rows.append({
                    "case_id": case["case_id"], "category": case["category"],
                    "method": method, "status": "no_results" if not candidates else "completed",
                    "failure_reason": None,
                    "result_note": (
                        f"{method} index is empty."
                        if not candidates and (
                            (method == "text" and text_count == 0)
                            or (method == "visual" and visual_count == 0)
                            or (method == "hybrid" and text_count == 0 and visual_count == 0)
                        )
                        else "No matching or valid assets returned."
                        if not candidates else None
                    ),
                    "ranking": ranked,
                    "relevant_ranked_ids": _relevant_ranked_ids(case, candidates),
                    "relevance": _case_relevance(case),
                })

    aggregates = []
    categories = sorted({case["category"] for case in eligible})
    for method in methods:
        for category in categories:
            subset = [row for row in rows if row["method"] == method and row["category"] == category]
            if not subset:
                continue
            rankings = [row["relevant_ranked_ids"] for row in subset]
            relevance = [row["relevance"] for row in subset]
            metric_values = evaluate_retrieval(rankings, relevance, k_values=k_values) if subset else {}
            aggregates.append({
                "method": method, "category": category, "case_count": len(subset),
                "valid_ground_truth": sum(bool(row.get("relevance")) for row in subset),
                "no_ground_truth": sum(not bool(row.get("relevance")) for row in subset),
                "no_results": sum(row["status"] == "no_results" for row in subset),
                "failed": sum(row["status"] == "failed" for row in subset),
                "metrics": metric_values,
            })
    return {
        "case_count": len(selected), "eligible_similarity_cases": len(eligible),
        "page_specific_controls": page_controls, "unlabelled_controls": unlabelled_controls,
        "non_real_controls": non_real_controls,
        "per_case": rows, "aggregates": aggregates,
        "k_values": list(k_values), "execution": "actual_existing_retrieval_pipeline_when_completed",
        "remote_text_embeddings_allowed": allow_remote,
    }


def _generation_input_experiment(
    dataset: dict[str, Any], *, split: str | None, allow_remote: bool,
    retrieval_fn=None, visual_fn=None, generate_fn=None,
) -> dict[str, Any]:
    """Compare text-only and same-text-plus-associated-images generation inputs."""
    from retrieval.embedder import get_embedder
    from retrieval.multi_vector_retriever import retrieve as real_retrieve
    from retrieval.visual_indexer import search_visual_assets as real_visual_search
    from retrieval.hybrid_retriever import select_hybrid_candidates
    from generation.rag_chain import generate_answer as real_generate_answer
    from evaluation.answer_metrics import evaluate_answer_case

    candidates = [
        case for case in dataset["cases"]
        if case.get("generation_evaluation")
        and (split is None or case["split"] == split)
    ]
    excluded = [
        {"case_id": case["case_id"], "status": "excluded_non_real_source"}
        for case in candidates
        if case.get("source", {}).get("kind") != "real_document"
    ]
    selected = [
        case for case in candidates
        if case.get("source", {}).get("kind") == "real_document"
    ]
    if not allow_remote:
        raise ValueError(
            "Generation-input evaluation also needs text embeddings. Pass "
            "--allow-remote-text-embeddings and --run-generation explicitly."
        )
    if not selected:
        return {
            "evaluation_kind": "mocked_test" if any((retrieval_fn, visual_fn, generate_fn)) else "actual_opt_in_model_generation",
            "case_count": len(candidates), "per_case": excluded, "generation_calls": 0,
            "cost": None, "note": "No real-document generation_evaluation cases selected.",
        }
    embedder = get_embedder()
    retrieval_fn = retrieval_fn or real_retrieve
    visual_fn = visual_fn or real_visual_search
    generate_fn = generate_fn or real_generate_answer
    injected = any((retrieval_fn is not real_retrieve, visual_fn is not real_visual_search, generate_fn is not real_generate_answer))
    results = []
    calls = 0
    for case in selected:
        case_id = case["case_id"]
        try:
            text_results = retrieval_fn(
                case["query"], embedder, k=5,
                document_id=case.get("source", {}).get("document_id"),
            )
            old_offline = os.environ.get("HF_HUB_OFFLINE")
            os.environ["HF_HUB_OFFLINE"] = "1"
            try:
                visual_results = visual_fn(
                    case["query"], limit=5,
                    document_id=case.get("source", {}).get("document_id"),
                )
            finally:
                if old_offline is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = old_offline
            fused = select_hybrid_candidates(
                text_results, visual_results, text_limit=5, visual_limit=5
            )
            text_context = [
                candidate for candidate in fused
                if "text" in candidate.get("retrieval_sources", {})
                and candidate.get("type") != "Image"
            ]
            if not text_context:
                results.append({
                    "case_id": case_id, "status": "no_text_evidence",
                    "failure_reason": "No textual evidence was retrieved; a controlled text versus text-plus-images comparison cannot be made.",
                })
                continue
            visual_candidates = [
                candidate for candidate in fused
                if "visual" in candidate.get("retrieval_sources", {})
            ]
            from main import _visual_evidence_items

            images = _visual_evidence_items(visual_candidates, text_context)
            if not images:
                results.append({
                    "case_id": case_id, "status": "no_image_evidence",
                    "failure_reason": "No associated available visual asset was retrieved; comparison was not run.",
                    "fixed_text_evidence_ids": [str(item.get("uuid")) for item in text_context],
                })
                continue
            # The exact same text items are passed in both conditions.
            calls += 1
            text_answer = generate_fn(case["query"], text_context)
            calls += 1
            multimodal_answer = generate_fn(case["query"], text_context + images)
            expected = case.get("generation_evaluation", {}).get("expected", {})
            results.append({
                "case_id": case_id, "status": "actual_model_generation",
                "text_only": {
                    "answer": text_answer,
                    **evaluate_answer_case(text_answer, expected),
                },
                "text_plus_images": {
                    "answer": multimodal_answer,
                    **evaluate_answer_case(multimodal_answer, expected),
                },
                "fixed_text_evidence_ids": [str(item.get("uuid")) for item in text_context],
                "image_evidence_ids": [str(item.get("uuid")) for item in images],
                "image_count": len(images),
            })
        except Exception as exc:
            results.append({
                "case_id": case_id, "status": "failed",
                "failure_reason": f"{type(exc).__name__}: {exc}",
            })
    return {
        "evaluation_kind": "mocked_pipeline_results" if injected else "actual_opt_in_model_generation",
        "case_count": len(candidates), "per_case": excluded + results, "generation_calls": calls,
        "cost": None,
        "note": "The same retrieved text context is supplied to both conditions; the second adds associated visual retrieval results. Cost is not estimated because provider usage data is not collected.",
    }


def build_report(
    mode: str,
    dataset: dict[str, Any] | None = None,
    *,
    split: str | None = None,
    allow_remote_text_embeddings: bool = False,
    k_values: tuple[int, ...] = DEFAULT_K,
) -> dict[str, Any]:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    report: dict[str, Any] = {
        "schema_version": 1,
        "run_id": timestamp.replace(":", "").replace("-", "").replace("+00:00", "Z"),
        "timestamp_utc": timestamp,
        "dataset_id": dataset.get("dataset_id") if dataset else None,
        "split": split,
        "source_kinds": (
            {
                kind: sum(case.get("source", {}).get("kind") == kind for case in dataset["cases"])
                for kind in sorted({case.get("source", {}).get("kind") for case in dataset["cases"]})
            }
            if dataset else {}
        ),
        "experiment": mode,
        "execution_kind": "synthetic_fixture_execution" if mode in {"all", "extraction", "answers"} else "real_pipeline_requested",
        "python_version": platform.python_version(),
        "case_count": sum(1 for case in dataset["cases"] if split is None or case["split"] == split) if dataset else 0,
        "config": {
            "text_embedding_model": os.getenv("EMBED_MODEL", "text-embedding-3-small"),
            "visual_embedding_model": os.getenv("VISUAL_EMBED_MODEL", "openai/clip-vit-base-patch32"),
            "text_collection": os.getenv("COLLECTION_NAME", "rag_documents"),
            "visual_collection": os.getenv("VISUAL_COLLECTION_NAME", "rag_visuals"),
            "qdrant_url_configured": bool(os.getenv("QDRANT_URL")),
            "k_values": list(k_values),
        },
        "limitations": [],
    }
    if mode in {"all", "retrieval"}:
        if dataset is None:
            report["retrieval"] = {
                "status": "pending_no_labelled_retrieval_dataset",
                "methods": ["text", "visual", "hybrid"],
                "per_case": [], "aggregates": [], "page_specific_controls": [],
                "eligible_similarity_cases": 0,
            }
            report["limitations"].append("No manually labelled, retrievable document dataset was supplied; no real retrieval quality metrics were calculated.")
        else:
            retrieval = _retrieval_experiment(dataset, split=split, allow_remote=allow_remote_text_embeddings, k_values=k_values)
            report["retrieval"] = retrieval
            if not allow_remote_text_embeddings:
                report["limitations"].append("Text and hybrid retrieval are not executed without explicit permission for remote text-embedding requests.")
            if not retrieval["eligible_similarity_cases"]:
                if retrieval["page_specific_controls"] and not retrieval["unlabelled_controls"]:
                    report["retrieval"]["status"] = "page_specific_controls_only"
                    report["limitations"].append("Only exact-page controls were present; page-specific retrieval bypasses similarity search by design.")
                elif retrieval["non_real_controls"] and not retrieval["unlabelled_controls"]:
                    report["retrieval"]["status"] = "pending_no_real_document_cases"
                    report["limitations"].append("Synthetic and mocked retrieval cases were not run against an unrelated real index; only manually labelled real-document cases are eligible.")
                else:
                    report["retrieval"]["status"] = "pending_no_labelled_retrieval_dataset"
                    report["limitations"].append("No similarity-search cases were eligible; page-specific cases bypass text/visual retrieval by design.")
    if mode in {"all", "extraction"}:
        from evaluation.structured_extraction_benchmark import run_benchmark

        report["extraction"] = run_benchmark()
        report["extraction"]["dataset_kind"] = "synthetic_generated_table_and_chart_fixtures"
        if mode == "extraction" and not dataset:
            report["case_count"] = report["extraction"]["fixture_case_count"]
        report["limitations"].append("Extraction metrics use controlled synthetic fixtures only; real-document extraction accuracy is unevaluated.")
    if mode in {"all", "answers"}:
        answer_cases = [] if dataset is None else [
            case for case in dataset["cases"]
            if case.get("answer_evaluation") and (split is None or case["split"] == split)
        ]
        results = []
        for case in answer_cases:
            fixture = case["answer_evaluation"]
            if "fixture_answer" not in fixture:
                results.append({"case_id": case["case_id"], "status": "missing_answer_input"})
                continue
            results.append({
                "case_id": case["case_id"], "status": "controlled_fixture_check",
                "execution_kind": fixture.get("execution_kind", "provided_fixture_answer"),
                **evaluate_answer_case(fixture["fixture_answer"], fixture["expected"]),
            })
        report["answers"] = {
            "evaluation_kind": "deterministic_controlled_fixture_checks",
            "case_count": len(answer_cases), "per_case": results,
            "passed": sum(item.get("passed", False) for item in results),
            "denominator": len(results),
            "required_fact_correct": sum(
                sum(result.get("required_facts", {}).values()) for result in results
            ),
            "required_fact_denominator": sum(
                len(result.get("required_facts", {})) for result in results
            ),
            "citation_correct": sum(
                sum(result.get("required_citations", {}).values()) for result in results
            ),
            "citation_denominator": sum(
                len(result.get("required_citations", {})) for result in results
            ),
            "missing_answer_inputs": sum(
                result.get("status") == "missing_answer_input" for result in results
            ),
        }
        if not answer_cases:
            report["limitations"].append("No labelled answer-evaluation cases were supplied; no answer-grounding accuracy was calculated.")
    if mode == "generation-input":
        if dataset is None:
            raise DatasetValidationError("Generation-input evaluation requires a labelled dataset.")
        report["generation_input"] = _generation_input_experiment(
            dataset, split=split, allow_remote=allow_remote_text_embeddings
        )
    if mode == "all":
        report["execution_kind"] = "synthetic_extraction_and_controlled_answer_fixtures"
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run reproducible PaperLens evaluation experiments.")
    parser.add_argument("--mode", choices=("all", "retrieval", "extraction", "answers", "generation-input"), default="all")
    parser.add_argument("--dataset", help="Version 1 labelled JSON dataset.")
    parser.add_argument("--split", help="Only evaluate cases in this split.")
    parser.add_argument("--output", help="JSON report path; a matching Markdown summary is also written.")
    parser.add_argument("--allow-remote-text-embeddings", action="store_true", help="Opt in to existing OpenAI text-embedding requests for retrieval evaluation.")
    parser.add_argument("--run-generation", action="store_true", help="Explicitly allow actual OpenAI generation calls in generation-input mode.")
    parser.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_K))
    args = parser.parse_args(argv)
    try:
        dataset = load_dataset(args.dataset) if args.dataset else None
        if args.k and any(k < 1 for k in args.k):
            raise DatasetValidationError("All retrieval cutoffs must be positive.")
        if args.mode == "generation-input" and not args.run_generation:
            raise DatasetValidationError(
                "generation-input mode makes actual model calls; pass --run-generation to opt in."
            )
        report = build_report(
            args.mode, dataset, split=args.split,
            allow_remote_text_embeddings=args.allow_remote_text_embeddings,
            k_values=tuple(args.k),
        )
        default = Path("evaluation/reports") / f"{report['run_id']}-{args.mode}.json"
        json_path, md_path = _write_report(report, args.output or default)
    except (DatasetValidationError, ValueError, FileExistsError) as exc:
        print(f"Invalid evaluation request: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Evaluation report: {json_path}")
    print(f"Markdown summary: {md_path}")
    print(f"Experiment: {report['experiment']} ({report['execution_kind']})")
    for limitation in report.get("limitations", []):
        print(f"Limitation: {limitation}")
    if args.mode == "retrieval":
        retrieval = report.get("retrieval", {})
        if retrieval.get("status") in {
            "pending_no_labelled_retrieval_dataset", "pending_no_real_document_cases",
            "page_specific_controls_only",
        } or not retrieval.get("eligible_similarity_cases"):
            return 1
        if any(row.get("status") == "failed" for row in retrieval.get("per_case", [])):
            return 1
    if args.mode == "generation-input":
        generation = report.get("generation_input", {})
        if not generation.get("case_count") or any(
            row.get("status") != "actual_model_generation"
            for row in generation.get("per_case", [])
        ):
            return 1
    if args.mode == "answers" and report.get("answers", {}).get("missing_answer_inputs"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
