"""Routing-vs-blending oracle decomposition over cached scores:
routing_value = oracle_selector − best_static_fusion;
blending_value = oracle_config − oracle_selector.
"""
from __future__ import annotations

from scaling_reranked_retrieval.application.derived import DerivedSearchAgent
from scaling_reranked_retrieval.domain.aggregate import qmean as _qmean
from scaling_reranked_retrieval.domain.conditions import CONDITIONS, SINGLETON_CONDITIONS
from scaling_reranked_retrieval.domain.metrics import CAP20_METRICS, metric as _metric

# The run harness's CONDITIONS menu minus the no-rerank hybrid baseline.
ORACLE_CONFIG_MENU = [c for c in CONDITIONS if c.name != "hybrid_only"]
SINGLETONS = [c for c in ORACLE_CONFIG_MENU if c.name in SINGLETON_CONDITIONS]
FUSION_CONFIGS = [c for c in ORACLE_CONFIG_MENU if c.name not in SINGLETON_CONDITIONS]

# One best-static-fusion blend, chosen by this metric, reused for every metric row.
SELECTION_METRIC = "recall_at_1"

METRIC_LABEL = {
    "recall_at_1": "R@1",
    "recall_at_5": "R@5",
    "recall_at_20": "R@20",
    "nDCG_at_10": "nDCG@10",
}


def per_query_condition_metrics(
    cache, qs, k: int, reranked_k: int, menu=ORACLE_CONFIG_MENU
) -> tuple[dict[str, dict[str, list[float]]], list[str]]:
    """Materialize every menu condition's metrics for every intersection query.

    Returns (table, queries): table[cond_name][metric] = per-query values
    aligned to `queries`. Pass a `menu` subset to restrict conditions.
    """
    queries = list(qs.gold.keys())
    table: dict[str, dict[str, list[float]]] = {
        c.name: {m: [] for m in CAP20_METRICS} for c in menu
    }
    for text in queries:
        gold_list = list(qs.gold[text])
        for cond in menu:
            agent = DerivedSearchAgent(
                cache=cache, retrieved_k=k, condition=cond, reranked_k=reranked_k
            )
            ranked = [o.object_id for o in agent.run(text)]
            for m in CAP20_METRICS:
                table[cond.name][m].append(_metric(m, gold_list, ranked))
    return table, queries


def _per_query_max(table: dict, names: list[str], metric: str, n: int) -> list[float]:
    """Per-query max of `metric` over the given condition names."""
    return [max(table[name][metric][i] for name in names) for i in range(n)]


def _decompose(
    table: dict, n: int, bsf_config: str, menu=ORACLE_CONFIG_MENU, singletons=SINGLETONS
) -> dict[str, dict[str, float]]:
    """Per-metric decomposition for one subset given a fixed best-static-fusion;
    pass restricted `menu`/`singletons` to decompose over a sub-menu."""
    singleton_names = [c.name for c in singletons]
    menu_names = [c.name for c in menu]
    out: dict[str, dict[str, float]] = {}
    for m in CAP20_METRICS:
        bsf = _qmean(table[bsf_config][m])
        sel = _qmean(_per_query_max(table, singleton_names, m, n))
        cfg = _qmean(_per_query_max(table, menu_names, m, n))
        out[METRIC_LABEL[m]] = {
            "best_static_fusion": bsf,
            "oracle_selector": sel,
            "oracle_config": cfg,
            "routing_value": sel - bsf,
            "blending_value": cfg - sel,
        }
    return out


def select_best_static_fusion(
    metric_values_by_config: dict[str, list[float]],
    fusion_configs=FUSION_CONFIGS,
) -> tuple[str, float]:
    """Argmax over the fusion blends of the MEAN of SELECTION_METRIC.

    Mean, not median (per-query-recall median is degenerate); ties broken by
    config name for determinism.
    """
    scored = [
        (_qmean(metric_values_by_config[c.name]), c.name)
        for c in fusion_configs
    ]
    best_mean, best_name = max(scored, key=lambda mv: (mv[0], _neg_name(mv[1])))
    return best_name, best_mean


def _neg_name(name: str) -> tuple:
    """Tie-break helper: prefer the alphabetically-first config on equal mean."""
    return tuple(-ord(c) for c in name)
