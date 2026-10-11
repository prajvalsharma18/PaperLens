"""
Unit tests for the RAGAS evaluator.
"""

import json
from unittest.mock import patch


def test_ragas_returns_three_metrics(tmp_path):
    """
    run_evaluation should return a dict with faithfulness,
    answer_relevancy, and context_precision keys.
    """
    from evaluation.ragas_evaluator import run_evaluation

    dataset = [
        {
            "question": "What is the revenue?",
            "answer": "Revenue was $10M.",
            "contexts": ["Revenue was $10M in Q3."],
            "ground_truth": "$10M",
        }
    ]

    dataset_path = tmp_path / "evaluation.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    mock_result = {
        "faithfulness": 0.92,
        "answer_relevancy": 0.88,
        "context_precision": 0.85,
    }

    with patch("ragas.evaluate", return_value=mock_result) as evaluate:
        scores = run_evaluation(str(dataset_path))

    assert set(scores.keys()) == {"faithfulness", "answer_relevancy", "context_precision"}
    assert scores == mock_result
    evaluate.assert_called_once()
    evaluated_dataset = evaluate.call_args.args[0]
    assert evaluated_dataset[0] == dataset[0]
    assert [metric.name for metric in evaluate.call_args.kwargs["metrics"]] == [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
    ]
