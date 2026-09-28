"""NDCG, scored the same way RankLib itself computes it — centralised here
so the trainer, the tuner and the evaluator all report the same number for
the same input.
"""
import math


def dcg_at_k(grades, k):
    """grades: relevance grades in ranked (not sorted) order."""
    total = 0.0
    for i, g in enumerate(grades[:k]):
        total += (2 ** g - 1) / math.log2(i + 2)
    return total


def ideal_dcg_at_k(grades, k):
    return dcg_at_k(sorted(grades, reverse=True), k)


def ndcg_at_k(grades, k):
    """grades: relevance grades of documents in the order they were ranked.
    Returns None when ideal DCG is 0 (no relevant doc in the list at all) —
    NDCG is undefined there, not zero. Callers decide whether to skip such
    queries or report them separately; silently defaulting to 0 or 1 both
    distort the headline average."""
    idcg = ideal_dcg_at_k(grades, k)
    if idcg == 0:
        return None
    return dcg_at_k(grades, k) / idcg
