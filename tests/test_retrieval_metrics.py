import pytest

from evaluation.retrieval_metrics import evaluate_retrieval


def test_evaluate_retrieval_computes_macro_recall_mrr_and_ndcg():
    scores = evaluate_retrieval(
        rankings=[["a", "x", "b"], ["x", "y"]],
        relevant_ids=[{"a", "b"}, {"z"}],
        k_values=(1, 2, 3),
    )

    assert scores["recall@1"] == 0.25
    assert scores["recall@2"] == 0.25
    assert scores["recall@3"] == 0.5
    assert scores["mrr"] == 0.5
    assert scores["ndcg@1"] == 0.5
    assert scores["ndcg@2"] == pytest.approx(0.3065735964)
    assert scores["ndcg@3"] == pytest.approx(0.4598, abs=0.001)


@pytest.mark.parametrize(
    ("rankings", "relevant_ids", "k_values", "message"),
    [
        ([["a"]], [], (1,), "corresponding"),
        ([], [], (1,), "At least one"),
        ([["a"]], [{"a"}], (), "positive integers"),
        ([["a"]], [{"a"}], (0,), "positive integers"),
        ([["a"]], [{"a"}], (1, 1), "duplicates"),
    ],
)
def test_evaluate_retrieval_rejects_invalid_labelled_inputs(
    rankings, relevant_ids, k_values, message
):
    with pytest.raises(ValueError, match=message):
        evaluate_retrieval(rankings, relevant_ids, k_values=k_values)


def test_no_ground_truth_scores_zero_and_remains_in_macro_denominator():
    scores = evaluate_retrieval([["hit"], ["anything"]], [{"hit"}, set()], k_values=(1,))
    assert scores["recall@1"] == 0.5
    assert scores["mrr"] == 0.5
    assert scores["ndcg@1"] == 0.5


def test_graded_ndcg_uses_gains_and_duplicate_results_keep_first_rank():
    scores = evaluate_retrieval(
        [["low", "low", "high"]],
        [{"high": 3, "low": 1}],
        k_values=(1, 2, 3),
    )
    assert scores["recall@3"] == 1.0
    assert scores["ndcg@1"] == 1 / 7
    assert scores["ndcg@3"] < 1.0
