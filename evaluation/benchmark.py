"""
Benchmark: Multimodal RAG vs Text-Only RAG.

Runs both pipelines against the same eval dataset and prints
a comparison table. Document the results in docs/evaluation-results.md.

Usage:
  python evaluation/benchmark.py --dataset data/eval_dataset.json
"""

import argparse
import json
from typing import Any


def run_text_only_rag(question: str, context: list[dict[str, Any]]) -> str:
    """
    Baseline: text-only RAG — ignores images, sends only text/table content.
    """
    from generation.rag_chain import generate_answer

    text_only_context = [
        item for item in context
        if item.get("type") != "Image"
    ]
    return generate_answer(question, text_only_context)


def run_multimodal_rag(question: str, context: list[dict[str, Any]]) -> str:
    from generation.rag_chain import generate_answer
    return generate_answer(question, context)


def benchmark(dataset_path: str) -> None:
    from retrieval.embedder import get_embedder
    from retrieval.multi_vector_retriever import retrieve
    from evaluation.ragas_evaluator import run_evaluation
    import tempfile
    import os

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset: list[dict[str, Any]] = json.load(f)

    embedder = get_embedder()

    multimodal_rows, text_only_rows = [], []

    for item in dataset:
        question = item["question"]
        ground_truth = item.get("ground_truth", "")
        context = retrieve(question, embedder)

        mm_answer = run_multimodal_rag(question, context)
        to_answer = run_text_only_rag(question, context)

        ctx_strings = [c.get("content", "") for c in context]

        multimodal_rows.append({
            "question": question,
            "answer": mm_answer,
            "contexts": ctx_strings,
            "ground_truth": ground_truth,
        })
        text_only_rows.append({
            "question": question,
            "answer": to_answer,
            "contexts": [c for c in ctx_strings if c],
            "ground_truth": ground_truth,
        })

    def _score(rows):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(rows, f)
            tmp_path = f.name
        try:
            return run_evaluation(tmp_path)
        finally:
            os.unlink(tmp_path)

    mm_scores = _score(multimodal_rows)
    to_scores = _score(text_only_rows)

    print("\n" + "=" * 55)
    print(f"{'Metric':<25} {'Multimodal':>12} {'Text-Only':>12}")
    print("=" * 55)
    for metric in ("faithfulness", "answer_relevancy", "context_precision"):
        print(f"{metric:<25} {mm_scores[metric]:>12.3f} {to_scores[metric]:>12.3f}")
    print("=" * 55)
    print("\nAdd these results to docs/evaluation-results.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    benchmark(args.dataset)
