"""Routing-vs-blending decomposition primitives (paper §5.1).

Per query the reranking optimum decomposes into ROUTING (picking the best
singleton per query) vs BLENDING (picking an interior fusion per query),
against a BEST-STATIC-FUSION reference (one fixed blend applied everywhere):

    routing_value  = oracle_selector − best_static_fusion
    blending_value = oracle_config   − oracle_selector

Shared by the oracle_config analysis (the production decomposition) and
noise_null (the same decomposition over noise-cloned caches).
"""
from __future__ import annotations

from src.application.derived import DerivedSearchAgent
from src.domain.aggregate import qmean as _qmean
from src.domain.conditions import CONDITIONS, SINGLETON_CONDITIONS
from src.domain.metrics import CAP20_METRICS, metric as _metric

# The oracle menu, matching the run harness exactly (the same CONDITIONS list
# run_search_eval consumes — equal-weight only), minus the no-rerank hybrid
# baseline.
ORACLE_CONFIG_MENU = [c for c in CONDITIONS if c.name != "hybrid_only"]
SINGLETONS = [c for c in ORACLE_CONFIG_MENU if c.name in SINGLETON_CONDITIONS]
FUSION_CONFIGS = [c for c in ORACLE_CONFIG_MENU if c.name not in SINGLETON_CONDITIONS]

# Single-config best-static-fusion selection metric (resolved spec decision:
# one blend, chosen by a primary metric, reused for every metric row).
SELECTION_METRIC = "recall_at_1"

# Pretty metric labels for the table / JSON (CAP20_METRICS order preserved).
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

    Returns (table, queries) where:
      table[cond_name][metric] = list of per-query values, aligned to `queries`.
    Each ranking is produced by DerivedSearchAgent at retrieved_k=k (RSF
    normalization recomputed on the restricted pool) and capped at reranked_k.

    `menu` defaults to the full ORACLE_CONFIG_MENU; pass a subset
    (e.g. RRF-only / RSF-only) to restrict which conditions are materialized.
    The noise-null experiment builds the full table once and decomposes it
    several ways via the `menu`/`singletons`/`fusion_configs` arguments on the
    functions below.
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
    """Per-metric decomposition for one subset given a fixed best-static-fusion.

    Returns {metric_label: {best_static_fusion, oracle_selector, oracle_config,
    routing_value, blending_value}}.

    `menu`/`singletons` default to the full menu and its three singletons;
    pass restricted lists (e.g. RRF-only or RSF-only) to decompose the same
    materialized `table` over a sub-menu.
    """
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

    `metric_values_by_config[name]` is the (pooled or per-subset) list of
    per-query SELECTION_METRIC values for that fusion config. Ties broken by
    config name for determinism. (Mean, not median: a per-query-recall median
    is degenerate — see src.aggregate — so a median-based argmax would tie
    dozens of blends at 0 and pick arbitrarily.)

    `fusion_configs` defaults to all equal-weight blends; pass the RRF-only or
    RSF-only blend list to select within a single fusion family (noise_null).
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
