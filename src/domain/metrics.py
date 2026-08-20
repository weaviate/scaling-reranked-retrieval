"""Metric computation — thin wrappers over the qab IR functions (exact match).

Using qab's own implementations guarantees analysis metrics match
run_search_eval byte-for-byte.
"""
from __future__ import annotations

from query_agent_benchmarking.internal.adapters.metrics.ir_metrics import (
    calculate_nDCG_at_k,
    calculate_recall_at_k,
)

from src.config import DatasetConfig

# Cap-20 metrics (directly comparable to runs/k{N}_from_k2000.json singletons).
CAP20_METRICS = ("recall_at_1", "recall_at_5", "recall_at_20", "nDCG_at_10")
# Extra metrics available only at output cap 100.
CAP100_METRICS = ("recall_at_50", "recall_at_100")

# Recall cutoffs added on top of the dataset's qab metrics profile (BRIGHT:
# recall@1/5/20 + nDCG@10; IRPAPERS: recall@1/5/20, with nDCG@10 supplied via
# DatasetConfig.extra_base_metrics). Cutoffs are filtered to retrieved_k at
# runtime so e.g. the k=200 run measures Recall@200 on hybrid_only (the
# first-stage ceiling). Reranked conditions only return reranked_k docs so any
# Recall@K with K>reranked_k caps at Recall@reranked_k there.
EXTRA_METRIC_CUTOFFS = (50, 100, 200, 500, 1000, 2000)


def metric(name: str, gold: list[str], ranked: list[str]) -> float:
    """Compute one named metric on a ranking using the qab implementations.

    `name` is a runs-file metric key like "recall_at_20" or "nDCG_at_10".
    The qab functions truncate internally to k, so a longer `ranked` is fine.
    """
    base, _, k_str = name.partition("_at_")
    k = int(k_str)
    if base == "recall":
        return float(calculate_recall_at_k(target_ids=gold, retrieved_ids=ranked, k=k))
    if base == "nDCG":
        return float(calculate_nDCG_at_k(target_ids=gold, retrieved_ids=ranked, k=k))
    raise ValueError(f"Unknown metric {name!r}")



def build_extra_metrics(retrieved_k: int, cfg: DatasetConfig) -> list[dict]:
    recall_cutoffs = [
        {"name": "recall", "params": {"k": k}}
        for k in EXTRA_METRIC_CUTOFFS
        if k <= retrieved_k
    ]
    return recall_cutoffs + list(cfg.extra_base_metrics)
