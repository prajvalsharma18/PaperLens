"""
RAGAS evaluation runner.

Loads an evaluation dataset and computes:
  - Faithfulness
  - Answer Relevancy
  - Context Precision

Dataset format (JSON list):
  [
    {
      "question": "What was Q3 revenue?",
      "answer": "...",          # ground truth answer
      "contexts": ["..."],     # retrieved context strings
      "ground_truth": "..."    # expected answer
    }
  ]
"""

import json
from typing import Any


def run_evaluation(dataset_path: str) -> dict[str, float]:
    """
    Run RAGAS metrics on a labelled evaluation dataset.
    Returns a dict mapping metric name to float score.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision

    with open(dataset_path, "r", encoding="utf-8") as f:
        raw: list[dict[str, Any]] = json.load(f)

    dataset = Dataset.from_list(raw)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )

    return {
        "faithfulness": result["faithfulness"],
        "answer_relevancy": result["answer_relevancy"],
        "context_precision": result["context_precision"],
    }
