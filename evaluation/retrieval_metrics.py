"""Offline ranking metrics for labelled retrieval evaluations."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def evaluate_retrieval(
    rankings: Sequence[Sequence[str]],
    relevant_ids: Sequence[Iterable[str]],
    *,
    k_values: Sequence[int] = (1, 3, 5),
) -> dict[str, float]:
    """Compute macro Recall@k and reciprocal rank over labelled query results."""
    if len(rankings) != len(relevant_ids):
        raise ValueError("Each ranking must have a corresponding relevant-ID set.")
    if not rankings:
        raise ValueError("At least one labelled query is required.")
    if not k_values or any(type(k) is not int or k < 1 for k in k_values):
        raise ValueError("k_values must contain positive integers.")
    if len(set(k_values)) != len(k_values):
        raise ValueError("k_values must not contain duplicates.")

    recalls = {k: 0.0 for k in k_values}
    reciprocal_rank_total = 0.0
    for ranking, relevant in zip(rankings, relevant_ids):
        relevant_set = set(relevant)
        if not relevant_set:
            raise ValueError("Each labelled query must have at least one relevant ID.")
        first_relevant_rank = next(
            (
                rank
                for rank, identifier in enumerate(ranking, start=1)
                if identifier in relevant_set
            ),
            None,
        )
        if first_relevant_rank is not None:
            reciprocal_rank_total += 1.0 / first_relevant_rank
        for k in k_values:
            retrieved_relevant = len(set(ranking[:k]) & relevant_set)
            recalls[k] += retrieved_relevant / len(relevant_set)

    query_count = len(rankings)
    return {
        **{
            f"recall@{k}": recall / query_count
            for k, recall in recalls.items()
        },
        "mrr": reciprocal_rank_total / query_count,
    }
