import pytest

from evaluation.retrieval_metrics import evaluate_retrieval


def test_evaluate_retrieval_computes_macro_recall_and_mrr():
    scores = evaluate_retrieval(
        rankings=[["a", "x", "b"], ["x", "y"]],
        relevant_ids=[{"a", "b"}, {"z"}],
        k_values=(1, 2, 3),
    )

    assert scores == {
        "recall@1": 0.25,
        "recall@2": 0.25,
        "recall@3": 0.5,
        "mrr": 0.5,
    }


@pytest.mark.parametrize(
    ("rankings", "relevant_ids", "k_values", "message"),
    [
        ([["a"]], [], (1,), "corresponding"),
        ([], [], (1,), "At least one"),
        ([["a"]], [{"a"}], (), "positive integers"),
        ([["a"]], [{"a"}], (0,), "positive integers"),
        ([["a"]], [{"a"}], (1, 1), "duplicates"),
        ([["a"]], [set()], (1,), "at least one relevant"),
    ],
)
def test_evaluate_retrieval_rejects_invalid_labelled_inputs(
    rankings, relevant_ids, k_values, message
):
    with pytest.raises(ValueError, match=message):
        evaluate_retrieval(rankings, relevant_ids, k_values=k_values)
