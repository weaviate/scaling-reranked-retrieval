"""Pure ranking/fusion functions over cached reranker scores.

This is the single implementation of the derive-time ranking semantics; the
DerivedSearchAgent and every analysis share it.

Behavior contract (do not "fix" without re-deriving every published number):
  - rsf_fuse iterates set(pool) when building the fused dict, so exact
    fused-score ties at the rank-K boundary break in string-hash order
    (PYTHONHASHSEED-dependent, ~1 query of wobble).
  - min_max is pool-dependent and must be recomputed on the restricted pool
    at every derive — never reused from a larger-k normalization.
  - All sorts are stable descending-by-score, so equal scores keep insertion
    order (hybrid-pool order for singletons, accumulation order for fusion).
"""
from __future__ import annotations

from src.config import RRF_K


def min_max(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def singleton_ranking(scores: dict[str, float], reranked_k: int) -> list[str]:
    """Rank one provider's pool-filtered scores; truncate to reranked_k."""
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [d for d, _ in ranked[:reranked_k]]


def rsf_fuse(
    per_reranker: dict[str, dict[str, float]],
    pool: list[str],
    weights: dict[str, float],
    rerankers: tuple[str, ...],
) -> dict[str, float]:
    """Relative Score Fusion: min-max normalize per reranker, weighted sum."""
    # Min-max normalize each reranker independently on the filtered pool.
    normed = {r: min_max(scores) for r, scores in per_reranker.items()}
    pool_set = set(pool)
    fused: dict[str, float] = {}
    for d in pool_set:
        fused[d] = sum(
            weights.get(r, 0.0) * normed[r].get(d, 0.0) for r in rerankers
        )
    return fused


def rrf_fuse(
    per_reranker: dict[str, dict[str, float]],
    weights: dict[str, float],
    rerankers: tuple[str, ...],
    rrf_k: int = RRF_K,
) -> dict[str, float]:
    """Reciprocal Rank Fusion over each reranker's score-sorted ranking."""
    fused: dict[str, float] = {}
    for r in rerankers:
        ranked_r = sorted(per_reranker[r].items(), key=lambda kv: kv[1], reverse=True)
        w = weights.get(r, 1.0 / len(rerankers))
        for rank, (d, _) in enumerate(ranked_r):
            fused[d] = fused.get(d, 0.0) + w / (rrf_k + rank + 1)
    return fused


def fused_ranking(fused: dict[str, float], reranked_k: int) -> list[str]:
    """Sort a fused-score dict descending; truncate to reranked_k."""
    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return [d for d, _ in ranked[:reranked_k]]


def rrf_fuse_rankings(
    rankings: dict[str, list[str]],
    weights: dict[str, float],
    rrf_k: int = RRF_K,
) -> list[str]:
    """RRF directly over ordered doc-id lists (rank-only rerankers, e.g. the
    listwise LLM rerankers, which emit rankings without scores).

    Same math as rrf_fuse (contribution w/(rrf_k + rank + 1)), but positions
    come straight from list order instead of a score sort. Exact fused-score
    ties (possible with symmetric weights) break deterministically by the
    first ranker's list order (stable sort over insertion order).
    """
    fused: dict[str, float] = {}
    for r, ranked in rankings.items():
        w = weights.get(r, 1.0 / len(rankings))
        for rank, d in enumerate(ranked):
            fused[d] = fused.get(d, 0.0) + w / (rrf_k + rank + 1)
    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return [d for d, _ in ordered]
