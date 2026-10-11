"""Deterministic ranking metrics for labelled retrieval evaluations."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence


def _unique(ranking: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for identifier in ranking:
        if identifier not in seen:
            seen.add(identifier)
            result.append(identifier)
    return result


def evaluate_retrieval(
    rankings: Sequence[Sequence[str]],
    relevant_ids: Sequence[Iterable[str] | Mapping[str, int]],
    *,
    k_values: Sequence[int] = (1, 3, 5),
) -> dict[str, float]:
    """Return macro Recall@k, MRR and nDCG@k.

    An empty relevance set is a labelled no-ground-truth case: it scores zero
    for every metric and remains in the macro denominator. For graded inputs,
    map evidence IDs to positive integer gains. Candidate rank order is trusted;
    tied ranks should be resolved by the retriever's stable output order.
    """
    if len(rankings) != len(relevant_ids):
        raise ValueError("Each ranking must have a corresponding relevance judgement.")
    if not rankings:
        raise ValueError("At least one labelled query is required.")
    if not k_values or any(type(k) is not int or k < 1 for k in k_values):
        raise ValueError("k_values must contain positive integers.")
    if len(set(k_values)) != len(k_values):
        raise ValueError("k_values must not contain duplicates.")

    recall = {k: 0.0 for k in k_values}
    ndcg = {k: 0.0 for k in k_values}
    reciprocal_rank = 0.0
    for ranking, judgement in zip(rankings, relevant_ids):
        ranked = _unique(ranking)
        if isinstance(judgement, Mapping):
            gains = {str(key): int(value) for key, value in judgement.items()}
        else:
            gains = {str(identifier): 1 for identifier in judgement}
        if any(type(gain) is not int or gain < 0 for gain in gains.values()):
            raise ValueError("Relevance grades must be non-negative integers.")
        gains = {identifier: gain for identifier, gain in gains.items() if gain > 0}
        if not gains:
            continue

        first = next(
            (rank for rank, identifier in enumerate(ranked, 1) if identifier in gains),
            None,
        )
        if first is not None:
            reciprocal_rank += 1.0 / first
        for k in k_values:
            recall[k] += len(set(ranked[:k]) & gains.keys()) / len(gains)
            dcg = sum(
                (2 ** gains.get(identifier, 0) - 1) / math.log2(rank + 1)
                for rank, identifier in enumerate(ranked[:k], 1)
            )
            ideal_gains = sorted(gains.values(), reverse=True)[:k]
            ideal = sum(
                (2 ** gain - 1) / math.log2(rank + 1)
                for rank, gain in enumerate(ideal_gains, 1)
            )
            ndcg[k] += dcg / ideal if ideal else 0.0

    count = len(rankings)
    return {
        **{f"recall@{k}": value / count for k, value in recall.items()},
        "mrr": reciprocal_rank / count,
        **{f"ndcg@{k}": value / count for k, value in ndcg.items()},
    }
