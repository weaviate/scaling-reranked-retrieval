"""Reranker agreement, overlap, and oracle-ceiling analysis.

Pure derivation over the existing k=2000 score caches — ZERO reranker API
calls (this module never imports a provider client; see acceptance criterion 1
in the spec). It measures the *mechanism* behind the mixture-of-rerankers
fusion lift: how decorrelated Cohere `rerank-v4.0-pro`, Voyage `rerank-2.5`,
and Zerank-2 actually are, how much headroom that decorrelation creates, and
what perfect per-query selection over the deployed fusion menu could score.

What it computes, per (dataset, retrieved_k):
  - Rank-1 agreement (pairwise + three-way)                     (spec 4.1)
  - Top-m Jaccard overlap, m in {5, 20, 100}                    (spec 4.2)
  - Full-pool Kendall tau-b rank correlation                   (spec 4.3)
  - R@1 correctness Venn decomposition (8 cells)               (spec 4.4)
  - Conditional top-1 precision (agree vs disagree)            (spec 4.5)
  - Oracle selector ceiling over {R@1,R@5,R@20,nDCG@10}        (spec 4.6)
  - Oracle-config@K — per-query oracle over the 29-condition  (spec 4.7)
    menu (3 singletons + 26 fusion blends), rank-respecting
All of the above at every k in {100,200,500,1000,2000}        (spec 4.8)

Why oracle-config and not oracle-merge: a set-union "oracle" credits a gold
doc the moment it appears anywhere in the union of three top-K lists — that
ignores rank order and overstates what any weighted fusion can realize (no
weighted fusion can keep one reranker's gold doc without inheriting whatever
that same weighting promotes above it). Oracle-config takes the per-query
max over the menu of actual reranked rankings, so each pick is a real
top-K block and respects rank order. It is the rank-respecting per-query
oracle over a deployable configuration set.

  oracle_config@K(q) = max over c in MENU of metric(top_K(ranking_c(q)))

Optional second pass (--simplex-grid): for each query, grid-sweep weights on
the 3-simplex (step 0.05 → 231 points), calling the retrieval-layer fuse_rsf and
fuse_rrf so fusion semantics match the experiment exactly. Per-query max
across the grid gives a LOWER bound on the continuous fusion-family
optimum (recall@K is piecewise-constant in weights; finite grids miss thin
optimal regions). Off by default — adds compute, still zero-API.

Metric functions are imported from query_agent_benchmarking so they match
run_search_eval exactly. Pool filtering + ranking reuse DerivedSearchAgent so
they are byte-identical to the derive path (regression guard, spec 9.3).

Outputs:
  results/bright_<dataset>/extras/agreement_k{N}_from_k2000.json  (one per k)
  results/AGREEMENT.md                                            (--all-datasets)

Usage:
    uv run python scripts/agreement.py --dataset biology
    uv run python scripts/agreement.py --dataset biology --retrieved-k 500
    uv run python scripts/agreement.py --all-datasets
    uv run python scripts/agreement.py --all-datasets --simplex-grid
    uv run python scripts/agreement.py --dataset biology --smoke
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from scipy.stats import kendalltau

from src.application.derived import DerivedSearchAgent
from src.adapters.cache import ScoreCache, validate_cache_for_use
from src.config import (
    CACHE_K,
    DATASETS,
    MODEL_OVERRIDES,
    PROV_LETTER,
    PROVIDERS,
    RANDOM_SEED,
    RESULTS_DIR,
    get_results_dir,
)
# Read-only re-use of the same Condition specs run_search_eval consumes; this
# guarantees oracle-config is consistent with runs/k{N}_from_k2000.json by
# construction.
from src.domain.conditions import (
    CONDITIONS as _RE_CONDITIONS,
    SINGLETON_CONDITIONS,
    _Condition,
)
from src.domain.metrics import CAP20_METRICS, metric as _metric
from src.application.queryset import QuerySet, build_query_set, load_and_validate

# Library fusion functions used only by the optional --simplex-grid pass.
# Importing them at module load (not under a flag) keeps the import path
# obvious; they're cheap and dependency-free of any client/credential.
from src.adapters.retrieval.rsf import fuse_rsf  # noqa: E402
from src.adapters.retrieval.rrf import fuse_rrf  # noqa: E402
from src.adapters.retrieval.models import RerankItem as _RerankItem  # noqa: E402

from src.adapters import qab

qab.setup()

# --------------------------------------------------------------------------- #
# Configuration (mirrors run_experiment.py)                                    #
# --------------------------------------------------------------------------- #

# DATASETS and get_results_dir are imported from src.config (above) so the
# (qab dataset, Weaviate collection, results dir) registry has a single source
# of truth. DATASETS values are DatasetConfig instances.

# Datasets with a reranked_k=100 sweep (extra R@50 / R@100 oracle + union).
RK100_DATASETS = {"biology", "psychology", "robotics", "irpapers_text"}

K_VALUES = (100, 200, 500, 1000, 2000)
RERANKED_K = 20  # deployed output cap; oracle/singletons comparable to runs/.

# Menu of reranked configurations over which oracle-config takes the per-query
# max. Currently the 29 conditions evaluated in run_search_eval — 3 singletons
# + 26 fusion blends — minus the no-rerank hybrid baseline. Widen by editing
# this one line; the analysis automatically adopts whatever conditions are in
# src.conditions.CONDITIONS (which is the same list run_search_eval consumes).
ORACLE_CONFIG_MENU: tuple = tuple(c for c in _RE_CONDITIONS if c.name != "hybrid_only")

# Optional --simplex-grid pass: weight-vector resolution on the 3-simplex.
# Step 0.05 → 21 ticks per axis → 231 weight vectors total (the triangular
# number C(22,2)). Halving the step quadruples grid size; the trade-off is
# wall-time vs how thin a recall@K plateau the grid can resolve.
SIMPLEX_STEP = 0.05
RRF_K0 = 60  # matches the constant baked into run_experiment / DerivedSearchAgent
RERANKED_K_100 = 100  # extended cap for R@50 / R@100 on RK100 datasets.
TOP_M = (5, 20, 100)

# Extra metrics available only at output cap 100.
CAP100_METRICS = ("recall_at_50", "recall_at_100")

BASELINE_CONDITION = "hybrid_only"


def _oracle_config_for_query(
    cache: ScoreCache,
    query: str,
    gold_list: list[str],
    retrieved_k: int,
    reranked_k_for_oc: int,
    metric_keys: Sequence[str],
) -> dict[str, float]:
    """Per-query oracle-config: max metric over ORACLE_CONFIG_MENU rankings.

    For each menu condition c, materialize the top-`reranked_k_for_oc` ranking
    via DerivedSearchAgent at the given retrieved_k, then compute every metric
    in metric_keys on that ranking and keep the per-metric max across conditions.
    Each pick is a rank-respecting top-K block from a real reranked configuration
    we actually ran — so the value is achievable by a perfect per-query
    selector over the menu (deployable in principle: train a classifier to
    select the config per query).

    qab's calculate_recall_at_k / nDCG_at_k slice internally to k, so a longer
    ranked list than the target metric K is fine.
    """
    per_metric_max: dict[str, float] = {m: -1.0 for m in metric_keys}
    for cond in ORACLE_CONFIG_MENU:
        agent = DerivedSearchAgent(
            cache=cache,
            retrieved_k=retrieved_k,
            condition=cond,
            reranked_k=reranked_k_for_oc,
        )
        ranked = [o.object_id for o in agent.run(query)]
        for m in metric_keys:
            v = _metric(m, gold_list, ranked)
            if v > per_metric_max[m]:
                per_metric_max[m] = v
    return per_metric_max


def _simplex_grid_points(step: float) -> list[tuple[float, float, float]]:
    """Enumerate (w_c, w_v, w_z) on the 3-simplex with the given step.

    Step 0.05 yields 231 points = C(22, 2). Each weight is a multiple of step
    in [0, 1], with the three summing exactly to 1 (modulo float rounding,
    which we resolve by rounding the third weight).
    """
    n_steps = int(round(1.0 / step))
    out: list[tuple[float, float, float]] = []
    for i in range(n_steps + 1):
        for j in range(n_steps + 1 - i):
            k = n_steps - i - j
            out.append((i * step, j * step, k * step))
    return out


def _rerank_items_from_pool(
    cache_entry: dict, pool: list[str], provider: str
) -> list[_RerankItem]:
    """Build the RerankItem list a library fuse_* function expects.

    `index` is a positional index into `pool` (the top-N hybrid order) so that
    the fused RerankItem.index can be resolved back to a doc-id via pool[idx].
    Providers with missing scores for some doc contribute that doc's `0.0`,
    which is what DerivedSearchAgent's inline fusion would also do.
    """
    scores = cache_entry.get(f"{provider}_scores", {}) or {}
    items: list[_RerankItem] = []
    for i, doc_id in enumerate(pool):
        s = scores.get(doc_id)
        if s is None:
            continue
        items.append(_RerankItem(index=i, relevance_score=float(s)))
    return items


def _simplex_grid_oracle(
    cache: ScoreCache,
    query: str,
    gold_list: list[str],
    retrieved_k: int,
    output_k: int,
    metric_keys: Sequence[str],
    step: float = SIMPLEX_STEP,
) -> dict[str, dict[str, float]]:
    """Per-query max over a 3-simplex grid of RSF and RRF weight vectors.

    Returns {"rsf": {metric: max}, "rrf": {metric: max}, "max": {metric: max}}.
    Calls the retrieval-layer fuse_rsf and fuse_rrf so fusion semantics match the
    experiment exactly. RSF normalization is implicit in fuse_rsf (it re-does
    min-max on the input set, which here is the top-`retrieved_k` pool). RRF
    uses RRF_K0 = 60 to match the experiment.
    """
    entry = cache.queries.get(query, {})
    pool = entry.get("hybrid_order", [])[:retrieved_k]
    if not pool:
        return {kind: {m: 0.0 for m in metric_keys} for kind in ("rsf", "rrf", "max")}

    # Pre-build per-provider RerankItem lists once; library fuse_* will slice
    # by union of indices internally.
    per_provider_items = {
        p: _rerank_items_from_pool(entry, pool, p) for p in PROVIDERS
    }

    out = {kind: {m: -1.0 for m in metric_keys} for kind in ("rsf", "rrf", "max")}

    for w_c, w_v, w_z in _simplex_grid_points(step):
        weights = {"cohere": w_c, "voyage": w_v, "zerank": w_z}
        # Skip degenerate all-zero (shouldn't happen with step=0.05 since one
        # weight will be 1.0 in that case, but defensive).
        if w_c + w_v + w_z == 0.0:
            continue
        for kind, fuser in (("rsf", lambda r, k, w: fuse_rsf(r, top_k=k, weights=w)),
                            ("rrf", lambda r, k, w: fuse_rrf(r, top_k=k, rrf_k=RRF_K0, weights=w))):
            fused = fuser(per_provider_items, output_k, weights)
            ranked = [pool[item.index] for item in fused]
            for m in metric_keys:
                v = _metric(m, gold_list, ranked)
                if v > out[kind][m]:
                    out[kind][m] = v
                if v > out["max"][m]:
                    out["max"][m] = v
    return out


# --------------------------------------------------------------------------- #
# Per-query ranking materialization (byte-identical to the derive path)        #
# --------------------------------------------------------------------------- #


def _full_rankings(
    cache: ScoreCache, query: str, k: int, present: list[str]
) -> dict[str, list[str]]:
    """Full ranking of the top-k hybrid pool per present provider.

    reranked_k is set to k so the agent returns the entire sorted pool (not the
    deployed top-20). Tie-break is (score desc, hybrid rank asc), exactly as in
    DerivedSearchAgent (stable sort over a pool-ordered dict).
    """
    out: dict[str, list[str]] = {}
    for r in present:
        agent = DerivedSearchAgent(
            cache=cache,
            retrieved_k=k,
            condition=_Condition(provider=r),
            reranked_k=k,
        )
        out[r] = [o.object_id for o in agent.run(query)]
    return out


# --------------------------------------------------------------------------- #
# Core statistics over the intersection set                                    #
# --------------------------------------------------------------------------- #


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    union = sa | sb
    return len(sa & sb) / len(union) if union else 1.0


def compute_k_stats(
    cache: ScoreCache, qs: QuerySet, k: int, include_cap100: bool,
    simplex_grid: bool = False,
) -> dict:
    """Compute every spec-4 statistic for one retrieved_k over the intersection.

    `simplex_grid=True` adds the optional second pass: per-query 3-simplex
    grid sweep over RSF and RRF weight vectors (231 points at step 0.05),
    reported as `oracle_fusion_grid` in the output.
    """
    queries = list(qs.gold.keys())
    n = len(queries)
    pairs = [("cohere", "voyage"), ("cohere", "zerank"), ("voyage", "zerank")]

    # Accumulators -------------------------------------------------------- #
    rank1_pair = {f"{a}_{b}": 0 for a, b in pairs}
    rank1_three = 0
    jaccard_sum = {m: {f"{a}_{b}": 0.0 for a, b in pairs} for m in TOP_M}
    jaccard_count = {m: 0 for m in TOP_M}
    tau_sum = {f"{a}_{b}": 0.0 for a, b in pairs}
    venn = {cell: 0 for cell in _venn_cells()}
    # Conditional precision buckets.
    agree_correct = 0
    agree_n = 0
    disagree_prec_sum = 0.0
    disagree_n = 0
    # Oracle accumulators.
    cap20 = list(CAP20_METRICS)
    oracle_sum = {mname: 0.0 for mname in cap20}
    singleton_sum_cap20 = {r: {mname: 0.0 for mname in cap20} for r in PROVIDERS}
    cap100 = list(CAP100_METRICS) if include_cap100 else []
    oracle_sum_cap100 = {mname: 0.0 for mname in cap100}
    singleton_sum_cap100 = {r: {mname: 0.0 for mname in cap100} for r in PROVIDERS}

    # Oracle-config accumulators (spec 4.7, replaces oracle-merge). Per query,
    # take max over the 29-condition menu of each condition's metric on its
    # rank-respecting top-K output. cap-20 metrics use a top-20 ranking; cap-100
    # metrics use a top-100 ranking (RK100 datasets only).
    oracle_config_sum_cap20 = {mname: 0.0 for mname in cap20}
    oracle_config_sum_cap100 = {mname: 0.0 for mname in cap100}
    reranked_for_oc_cap20 = min(RERANKED_K, k)
    reranked_for_oc_cap100 = min(RERANKED_K_100, k)

    # Optional simplex-grid accumulators (off unless simplex_grid=True).
    fg_metric_keys = cap20 + (cap100 if include_cap100 else [])
    fg_kinds = ("rsf", "rrf", "max")
    # For simplex grid the output cap matches oracle-config so the two are
    # directly comparable: top-100 on RK100, top-20 otherwise.
    fg_output_k = reranked_for_oc_cap100 if include_cap100 else reranked_for_oc_cap20
    fusion_grid_sum: Optional[dict[str, dict[str, float]]] = (
        {kind: {m: 0.0 for m in fg_metric_keys} for kind in fg_kinds}
        if simplex_grid else None
    )

    for text in queries:
        gold = qs.gold[text]
        gold_list = list(gold)
        rankings = _full_rankings(cache, text, k, list(PROVIDERS))

        top1 = {r: rankings[r][0] for r in PROVIDERS}

        # 4.1 rank-1 agreement
        for a, b in pairs:
            if top1[a] == top1[b]:
                rank1_pair[f"{a}_{b}"] += 1
        all_agree = top1["cohere"] == top1["voyage"] == top1["zerank"]
        if all_agree:
            rank1_three += 1

        # 4.2 top-m Jaccard (skip m > k)
        for m in TOP_M:
            if m > k:
                continue
            jaccard_count[m] += 1
            for a, b in pairs:
                jaccard_sum[m][f"{a}_{b}"] += _jaccard(
                    rankings[a][:m], rankings[b][:m]
                )

        # 4.3 full-pool Kendall tau-b. A provider's score map can occasionally
        # cover a slightly different doc set than the hybrid pool (a chunk
        # boundary or per-doc filter drops one), so align over the docs ALL
        # THREE scored, taken in stable hybrid-rank order. In the common case
        # this is the entire pool.
        scored_sets = {r: set(rankings[r]) for r in PROVIDERS}
        common = scored_sets["cohere"] & scored_sets["voyage"] & scored_sets["zerank"]
        common_ordered = [d for d in rankings["cohere"] if d in common]
        score_arrays = {
            r: [cache.queries[text][f"{r}_scores"][d] for d in common_ordered]
            for r in PROVIDERS
        }
        for a, b in pairs:
            tau = float(kendalltau(score_arrays[a], score_arrays[b]).correlation)
            # kendalltau returns nan only for degenerate (constant) inputs;
            # treat that as zero correlation so the mean stays well-defined.
            tau_sum[f"{a}_{b}"] += 0.0 if tau != tau else tau

        # 4.4 Venn: subset of providers whose top-1 is in gold
        correct = frozenset(r for r in PROVIDERS if top1[r] in gold)
        venn[_venn_key(correct)] += 1

        # 4.5 conditional top-1 precision
        if all_agree:
            agree_n += 1
            if top1["cohere"] in gold:
                agree_correct += 1
        else:
            disagree_n += 1
            disagree_prec_sum += sum(1 for r in PROVIDERS if top1[r] in gold) / len(
                PROVIDERS
            )

        # 4.6 oracle selector + per-singleton metrics at cap 20
        for mname in cap20:
            per_r = {r: _metric(mname, gold_list, rankings[r][:RERANKED_K]) for r in PROVIDERS}
            oracle_sum[mname] += max(per_r.values())
            for r in PROVIDERS:
                singleton_sum_cap20[r][mname] += per_r[r]

        # cap-100 singleton + oracle metrics (RK100 datasets only).
        if include_cap100:
            for mname in cap100:
                per_r = {
                    r: _metric(mname, gold_list, rankings[r][:RERANKED_K_100])
                    for r in PROVIDERS
                }
                oracle_sum_cap100[mname] += max(per_r.values())
                for r in PROVIDERS:
                    singleton_sum_cap100[r][mname] += per_r[r]

        # 4.7 oracle-config — per-query max over the ORACLE_CONFIG_MENU of each
        # menu condition's metric on its rank-respecting top-K ranking. Each
        # menu pick is a real reranked configuration we ran, so the ceiling is
        # achievable by a perfect per-query selector over the menu (whereas
        # oracle-merge's union credit was not — see module docstring).
        oc_cap20 = _oracle_config_for_query(
            cache, text, gold_list, k, reranked_for_oc_cap20, cap20
        )
        for mname in cap20:
            oracle_config_sum_cap20[mname] += oc_cap20[mname]
        if include_cap100:
            oc_cap100 = _oracle_config_for_query(
                cache, text, gold_list, k, reranked_for_oc_cap100, cap100
            )
            for mname in cap100:
                oracle_config_sum_cap100[mname] += oc_cap100[mname]

        # Optional simplex-grid: per-query best over 231 weight vectors × {RSF, RRF}.
        if simplex_grid and fusion_grid_sum is not None:
            grid_max = _simplex_grid_oracle(
                cache, text, gold_list, k, fg_output_k, fg_metric_keys
            )
            for kind in fg_kinds:
                for m in fg_metric_keys:
                    fusion_grid_sum[kind][m] += grid_max[kind][m]

    # Assemble means ------------------------------------------------------ #
    def mean(x: float) -> float:
        return x / n if n else 0.0

    stats: dict = {
        "num_queries_intersection": n,
        "rank1_agreement": {
            **{pair: mean(c) for pair, c in rank1_pair.items()},
            "three_way": mean(rank1_three),
        },
        "topm_jaccard": {
            str(m): {
                pair: (jaccard_sum[m][pair] / jaccard_count[m])
                for pair in jaccard_sum[m]
            }
            for m in TOP_M
            if jaccard_count[m] > 0
        },
        "kendall_tau": {pair: mean(s) for pair, s in tau_sum.items()},
        "venn": _venn_report(venn, n),
        "conditional_top1_precision": {
            "agree": (agree_correct / agree_n) if agree_n else None,
            "disagree": (disagree_prec_sum / disagree_n) if disagree_n else None,
            "n_agree": agree_n,
            "n_disagree": disagree_n,
            "note": (
                "agree = all three top-1 identical; precision is P(common top-1 "
                "in gold). disagree precision is the mean over the three "
                "rerankers of 1[top-1 in gold], averaged over disagreeing queries."
            ),
        },
        "oracle_selector": {f"avg_{m}": mean(oracle_sum[m]) for m in cap20},
        "oracle_config": {f"avg_{m}": mean(oracle_config_sum_cap20[m]) for m in cap20},
        "singleton_metrics_cap20": {
            r: {f"avg_{m}": mean(singleton_sum_cap20[r][m]) for m in cap20}
            for r in PROVIDERS
        },
    }
    # Fusion headroom + unreachable (derived from Venn).
    stats["venn"]["fusion_headroom"] = mean(
        sum(venn[k_] for k_ in venn if 0 < len(_venn_unkey(k_)) < len(PROVIDERS))
    )
    stats["venn"]["unreachable"] = stats["venn"]["fractions"]["none"]

    if include_cap100:
        stats["oracle_selector_cap100"] = {
            f"avg_{m}": mean(oracle_sum_cap100[m]) for m in cap100
        }
        stats["oracle_config_cap100"] = {
            f"avg_{m}": mean(oracle_config_sum_cap100[m]) for m in cap100
        }
        stats["singleton_metrics_cap100"] = {
            r: {f"avg_{m}": mean(singleton_sum_cap100[r][m]) for m in cap100}
            for r in PROVIDERS
        }
    if simplex_grid and fusion_grid_sum is not None:
        stats["oracle_fusion_grid"] = {
            "step": SIMPLEX_STEP,
            "rrf_k0": RRF_K0,
            "output_k": fg_output_k,
            "metrics": {
                kind: {f"avg_{m}": mean(fusion_grid_sum[kind][m]) for m in fg_metric_keys}
                for kind in fg_kinds
            },
        }
    return stats


# --------------------------------------------------------------------------- #
# Venn helpers                                                                  #
# --------------------------------------------------------------------------- #


def _venn_cells() -> list[str]:
    """Eight Venn cell keys: none, c, v, z, cv, cz, vz, cvz (PROVIDERS order)."""
    letters = [PROV_LETTER[p] for p in PROVIDERS]
    cells = ["none"]
    # singletons
    cells += letters
    # pairs
    cells += ["".join([letters[i], letters[j]]) for i in range(3) for j in range(i + 1, 3)]
    # triple
    cells += ["".join(letters)]
    return cells


def _venn_key(correct: frozenset) -> str:
    if not correct:
        return "none"
    return "".join(PROV_LETTER[p] for p in PROVIDERS if p in correct)


def _venn_unkey(key: str) -> set[str]:
    if key == "none":
        return set()
    inv = {v: k for k, v in PROV_LETTER.items()}
    return {inv[ch] for ch in key}


def _venn_report(venn: dict[str, int], n: int) -> dict:
    return {
        "counts": dict(venn),
        "fractions": {cell: (venn[cell] / n if n else 0.0) for cell in venn},
    }


# --------------------------------------------------------------------------- #
# Comparison values from existing runs/ files (self-containment, spec 7)        #
# --------------------------------------------------------------------------- #


def find_runs_file(dataset_slug: str, k: int) -> Optional[Path]:
    """Locate the runs summary for (dataset, k), preferring a from_k2000 file.

    R@1/R@5/R@20/nDCG@10 are byte-identical across cache provenance and output
    cap, so any matching file serves for those; rk100 files additionally carry
    R@50 / R@100. Preference order keeps the regression guard pointed at the
    from_k2000 numbers the spec names.
    """
    rd = get_results_dir(dataset_slug)
    candidates = [
        rd / "runs" / f"k{k}_from_k{CACHE_K}.json",
        rd / "runs_rk100" / f"k{k}_from_k{CACHE_K}.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fall back to any provenance for this k.
    for sub in ("runs", "runs_rk100"):
        matches = sorted((rd / sub).glob(f"k{k}_from_k*.json")) if (rd / sub).exists() else []
        if matches:
            return matches[0]
    return None


def extract_comparison(runs_path: Optional[Path], include_cap100: bool) -> dict:
    """Pull hybrid baseline, best singleton, and best fusion per metric."""
    metrics = list(CAP20_METRICS) + (list(CAP100_METRICS) if include_cap100 else [])
    if runs_path is None or not runs_path.exists():
        return {"source": None}
    with open(runs_path) as f:
        payload = json.load(f)
    results = payload.get("results", {})

    def value(cond: str, metric: str) -> Optional[float]:
        entry = results.get(cond)
        if not entry or "error" in entry:
            return None
        return entry.get(f"avg_{metric}_mean")

    comparison: dict = {"source": str(runs_path)}
    comparison["hybrid_only"] = {m: value(BASELINE_CONDITION, m) for m in metrics}
    fusion_conditions = [
        c for c in results
        if c not in SINGLETON_CONDITIONS and c != BASELINE_CONDITION
    ]
    best_singleton: dict = {}
    best_fusion: dict = {}
    def best(conditions: list[str], m: str) -> Optional[dict]:
        scored = [(c, v) for c in conditions if (v := value(c, m)) is not None]
        if not scored:
            return None
        cond, val = max(scored, key=lambda kv: kv[1])
        return {"condition": cond, "value": val}

    for m in metrics:
        best_singleton[m] = best(list(SINGLETON_CONDITIONS), m)
        best_fusion[m] = best(fusion_conditions, m)
    comparison["best_singleton"] = best_singleton
    comparison["best_fusion"] = best_fusion
    return comparison


# --------------------------------------------------------------------------- #
# Regression guard + invariants (spec 9)                                        #
# --------------------------------------------------------------------------- #


def regression_singletons(cache: ScoreCache, qs: QuerySet, k: int) -> dict:
    """Recompute each singleton's cap-20 metrics over that provider's OWN set.

    This reproduces what run_search_eval evaluated (full dataset, queries where
    the provider's score map is missing get dropped), so the numbers should
    match runs/k{N}_from_k2000.json exactly (spec 9.3).
    """
    out: dict[str, dict] = {}
    for r in PROVIDERS:
        own = qs.present_by_provider[r]
        n = len(own)
        sums = {m: 0.0 for m in CAP20_METRICS}
        for text, gold in own.items():
            agent = DerivedSearchAgent(
                cache=cache, retrieved_k=k, condition=_Condition(provider=r),
                reranked_k=RERANKED_K,
            )
            ranked = [o.object_id for o in agent.run(text)]
            gl = list(gold)
            for m in CAP20_METRICS:
                sums[m] += _metric(m, gl, ranked)
        out[r] = {
            "n_queries": n,
            **{f"avg_{m}": (sums[m] / n if n else 0.0) for m in CAP20_METRICS},
        }
    return out


def _drop_tolerance(stats: dict) -> float:
    """Max-bias tolerance for denominator mismatch between runs/ best_fusion
    (averaged over each condition's own present-query set, up to len(cache)
    queries) and oracle_config (averaged over the all-three-present intersection
    set). Equals max_dropped / n_intersection — the biggest possible bias if
    every dropped query had recall 1.0 on the runs/ side.

    `stats` carries `_n_drops_max` set by analyze_dataset_k.
    """
    n_int = stats.get("num_queries_intersection", 0)
    max_drop = stats.get("_n_drops_max", 0)
    return (max_drop / n_int) if n_int > 0 else 0.0


def check_invariants(
    stats: dict,
    include_cap100: bool,
    comparison: Optional[dict] = None,
    prior_oracle_merge: Optional[dict] = None,
) -> list[str]:
    """Return a list of invariant-violation messages (empty == all pass).

    `comparison` carries best-fusion values from runs/ for the oracle-config >=
    best-fusion invariant; if absent, that check is skipped. `prior_oracle_merge`
    carries the previous oracle-merge values read from the existing JSON (before
    overwrite) for the one-time regression sanity bound; absent on fresh
    installs.
    """
    errs: list[str] = []
    n = stats["num_queries_intersection"]

    # All agreement values in [0, 1].
    for pair, v in stats["rank1_agreement"].items():
        if not (0.0 <= v <= 1.0):
            errs.append(f"rank1_agreement[{pair}]={v} out of [0,1]")
    for m, pd in stats["topm_jaccard"].items():
        for pair, v in pd.items():
            if not (0.0 <= v <= 1.0):
                errs.append(f"jaccard@{m}[{pair}]={v} out of [0,1]")

    # Venn cells sum to the intersection query count.
    cell_total = sum(stats["venn"]["counts"].values())
    if cell_total != n:
        errs.append(f"venn cells sum {cell_total} != intersection n {n}")

    # Oracle-selector R@1 == 1 - venn[none] exactly (spec 9.2, unchanged).
    oracle_r1 = stats["oracle_selector"]["avg_recall_at_1"]
    none_frac = stats["venn"]["fractions"]["none"]
    if abs(oracle_r1 - (1.0 - none_frac)) > 1e-9:
        errs.append(
            f"oracle_selector R@1 {oracle_r1} != 1 - venn[none] {1.0 - none_frac}"
        )

    # Oracle-selector >= every singleton on every cap-20 metric.
    for m in CAP20_METRICS:
        omet = stats["oracle_selector"][f"avg_{m}"]
        for r in PROVIDERS:
            smet = stats["singleton_metrics_cap20"][r][f"avg_{m}"]
            if omet < smet - 1e-9:
                errs.append(f"oracle_selector {m} {omet} < singleton {r} {smet}")

    # Oracle-config@K >= oracle-selector@K — the singletons are in the menu, so
    # the per-query max over the menu can only dominate the per-query best
    # singleton. (Equality when the per-query best is always a pure singleton.)
    for m in CAP20_METRICS:
        oc = stats["oracle_config"][f"avg_{m}"]
        os_ = stats["oracle_selector"][f"avg_{m}"]
        if oc < os_ - 1e-9:
            errs.append(f"oracle_config {m} {oc} < oracle_selector {m} {os_}")

    # Oracle-config@K >= best static fusion — a single fixed fusion config is
    # one of the menu picks for every query, so picking per-query can only
    # dominate the population-best static config. Caveat: the runs/ best_fusion
    # averages over each condition's own present-query set (the queries where
    # that condition could be scored), but oracle_config averages over the
    # all-three-present intersection. A no-{dropped-provider} fusion sees more
    # queries than the intersection, which can put it above oracle_config purely
    # from denominator mismatch. We absorb that with a tolerance equal to the
    # max-drop fraction (the largest possible bias from N missing queries).
    if comparison and comparison.get("best_fusion"):
        denom_tol = _drop_tolerance(stats)
        for m in CAP20_METRICS:
            bf = comparison["best_fusion"].get(m)
            if not bf:
                continue
            oc = stats["oracle_config"][f"avg_{m}"]
            if oc < bf["value"] - denom_tol - 1e-9:
                errs.append(
                    f"oracle_config {m} {oc} < best static fusion {bf['condition']} {bf['value']} (denom_tol={denom_tol:.4f})"
                )

    # Regression sanity (one-time, this rewrite only). A rank-respecting
    # selection cannot exceed the loose set-union ceiling that the prior
    # oracle-merge computation reported at the same cell. We log where this
    # binds (oracle-config equal to or within noise of the old ceiling), since
    # that signals a query where the rank-respecting pick exhausted the
    # diversity that the union credit was reading. Absence of priors is fine
    # (fresh install).
    if prior_oracle_merge:
        oc20 = stats["oracle_config"]["avg_recall_at_20"]
        pom20 = prior_oracle_merge.get("at_20")
        if pom20 is not None and oc20 > pom20 + 1e-9:
            errs.append(
                f"oracle_config R@20 {oc20} > prior oracle_merge@20 {pom20}"
            )

    if include_cap100:
        # Oracle-selector >= cap-100 singletons.
        for m in CAP100_METRICS:
            omet = stats["oracle_selector_cap100"][f"avg_{m}"]
            for r in PROVIDERS:
                smet = stats["singleton_metrics_cap100"][r][f"avg_{m}"]
                if omet < smet - 1e-9:
                    errs.append(
                        f"oracle_selector_cap100 {m} {omet} < singleton {r} {smet}"
                    )
        # Oracle-config@{50,100} >= oracle-selector@{50,100}.
        for m in CAP100_METRICS:
            oc = stats["oracle_config_cap100"][f"avg_{m}"]
            os_ = stats["oracle_selector_cap100"][f"avg_{m}"]
            if oc < os_ - 1e-9:
                errs.append(
                    f"oracle_config_cap100 {m} {oc} < oracle_selector_cap100 {m} {os_}"
                )
        # Oracle-config_cap100 >= best static fusion at cap-100 metrics; same
        # denominator-mismatch tolerance as the cap-20 case above.
        if comparison and comparison.get("best_fusion"):
            denom_tol = _drop_tolerance(stats)
            for m in CAP100_METRICS:
                bf = comparison["best_fusion"].get(m)
                if not bf:
                    continue
                oc = stats["oracle_config_cap100"][f"avg_{m}"]
                if oc < bf["value"] - denom_tol - 1e-9:
                    errs.append(
                        f"oracle_config_cap100 {m} {oc} < best fusion {bf['condition']} {bf['value']} (denom_tol={denom_tol:.4f})"
                    )
        # Regression sanity at cap 100 (one-time).
        if prior_oracle_merge:
            for K, mname in zip((50, 100), CAP100_METRICS):
                pomK = prior_oracle_merge.get(f"at_{K}")
                if pomK is None:
                    continue
                ocK = stats["oracle_config_cap100"][f"avg_{mname}"]
                if ocK > pomK + 1e-9:
                    errs.append(
                        f"oracle_config_cap100 {mname} {ocK} > prior oracle_merge@{K} {pomK}"
                    )

    # Optional simplex-grid invariant: grid max >= best static fusion.
    if "oracle_fusion_grid" in stats and comparison and comparison.get("best_fusion"):
        max_metrics = stats["oracle_fusion_grid"]["metrics"]["max"]
        for m in CAP20_METRICS:
            bf = comparison["best_fusion"].get(m)
            if not bf:
                continue
            gv = max_metrics.get(f"avg_{m}")
            if gv is not None and gv < bf["value"] - 1e-9:
                errs.append(
                    f"oracle_fusion_grid[max] {m} {gv} < best static fusion {bf['condition']} {bf['value']}"
                )

    return errs


# --------------------------------------------------------------------------- #
# Driver: one (dataset, k) cell                                                #
# --------------------------------------------------------------------------- #


def _read_prior_oracle_merge(dataset_slug: str, k: int) -> Optional[dict]:
    """Read prior oracle_merge values from the existing JSON file (if any).

    Used for the one-time regression sanity check oracle-config <= prior
    oracle-merge. After this run overwrites the file the values are gone, but
    they served their purpose. Returns a dict keyed by f"at_{K}" or None.
    """
    path = (
        get_results_dir(dataset_slug)
        / "extras"
        / f"agreement_k{k}_from_k{CACHE_K}.json"
    )
    if not path.exists():
        return None
    try:
        with open(path) as f:
            prev = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    out: dict[str, float] = {}
    om = prev.get("oracle_merge", {}) or {}
    if "at_20" in om:
        out["at_20"] = float(om["at_20"])
    om100 = prev.get("oracle_merge_cap100", {}) or {}
    for K in (50, 100):
        key = f"at_{K}"
        if key in om100:
            out[key] = float(om100[key])
    return out or None


def _log_regression_tightness(
    dataset_slug: str, k: int, stats: dict, prior_oracle_merge: Optional[dict]
) -> None:
    """Print where oracle-config equals (within noise) the prior oracle-merge."""
    if not prior_oracle_merge:
        return
    eps = 1e-6
    binds: list[str] = []
    oc20 = stats["oracle_config"]["avg_recall_at_20"]
    pom20 = prior_oracle_merge.get("at_20")
    if pom20 is not None and abs(oc20 - pom20) < eps:
        binds.append(f"R@20: oc={oc20:.4f} == prior om={pom20:.4f}")
    if "oracle_config_cap100" in stats:
        for K, mname in zip((50, 100), CAP100_METRICS):
            pK = prior_oracle_merge.get(f"at_{K}")
            if pK is None:
                continue
            ocK = stats["oracle_config_cap100"][f"avg_{mname}"]
            if abs(ocK - pK) < eps:
                binds.append(f"{mname}: oc={ocK:.4f} == prior om={pK:.4f}")
    if binds:
        print(f"  [regression] {dataset_slug} k={k}: oracle-config bound by prior oracle-merge at: {', '.join(binds)}")


def analyze_dataset_k(
    dataset_slug: str,
    cache: ScoreCache,
    qs: QuerySet,
    k: int,
    write: bool = True,
    simplex_grid: bool = False,
    read_prior: bool = True,
) -> dict:
    """Run one (dataset, k) cell.

    `read_prior=False` disables the regression-vs-prior-oracle-merge sanity
    check. The prior values are aggregates over the full intersection query
    set, so they are not comparable when analyzing a subset (smoke test).
    """
    _cfg = DATASETS[dataset_slug]
    dataset_name, collection_name = _cfg.qab_name, _cfg.collection
    include_cap100 = dataset_slug in RK100_DATASETS

    # Read prior oracle_merge values BEFORE overwriting the JSON, for the
    # one-time regression sanity check.
    prior_oracle_merge = _read_prior_oracle_merge(dataset_slug, k) if read_prior else None

    stats = compute_k_stats(cache, qs, k, include_cap100, simplex_grid=simplex_grid)
    # Denominator-mismatch tolerance for oracle_config >= best_fusion: the
    # largest single-provider drop count, expressed as a fraction of the
    # intersection size. Carried in stats so check_invariants can read it
    # without re-plumbing qs through every helper.
    stats["_n_drops_max"] = max((len(v) for v in qs.drops.values()), default=0)
    regression = regression_singletons(cache, qs, k)
    runs_path = find_runs_file(dataset_slug, k)
    comparison = extract_comparison(runs_path, include_cap100)
    invariant_errors = check_invariants(
        stats, include_cap100, comparison=comparison,
        prior_oracle_merge=prior_oracle_merge,
    )
    # Strip the internal field — not part of the public JSON schema.
    stats.pop("_n_drops_max", None)
    _log_regression_tightness(dataset_slug, k, stats, prior_oracle_merge)

    payload = {
        "dataset": dataset_name,
        "dataset_slug": dataset_slug,
        "collection": collection_name,
        "retrieved_k": k,
        "cache_retrieved_k": CACHE_K,
        "reranked_k": RERANKED_K,
        "model_overrides": MODEL_OVERRIDES,
        "random_seed": RANDOM_SEED,
        "drops": qs.drops,
        "dropped_total": sum(len(v) for v in qs.drops.values()),
        "queries_without_gold": qs.no_gold,
        **stats,
        "singleton_regression_own_set": regression,
        "comparison_from_runs": comparison,
        "invariant_errors": invariant_errors,
        "invariants_ok": not invariant_errors,
    }

    if write:
        out_path = (
            get_results_dir(dataset_slug)
            / "extras"
            / f"agreement_k{k}_from_k{CACHE_K}.json"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  wrote {out_path}")
    return payload


# --------------------------------------------------------------------------- #
# AGREEMENT.md report (spec 7)                                                  #
# --------------------------------------------------------------------------- #


def _f(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _cell(d: Optional[dict]) -> str:
    """Format a {condition, value} comparison cell."""
    if not d:
        return "n/a"
    return f"{d['value']:.3f} (`{d['condition']}`)"


def write_agreement_md(all_payloads: dict[str, dict[int, dict]]) -> Path:
    """Regenerate the cross-domain AGREEMENT.md from collected per-(ds,k) data."""
    datasets = [d for d in DATASETS if d in all_payloads and all_payloads[d]]
    lines: list[str] = []
    A = lines.append

    A("# Reranker Agreement, Overlap, and Oracle Ceiling")
    A("")
    A(
        "Cross-domain analysis of how decorrelated Cohere `rerank-v4.0-pro`, "
        "Voyage `rerank-2.5`, and Zerank-2 are, and the headroom that "
        "decorrelation creates. All numbers derived from the k=2000 score "
        "caches with zero reranker-API calls. Statistics are computed over the "
        "intersection of queries scored by all three providers (per-dataset "
        "drop counts in each `extras/agreement_k{N}_from_k2000.json`)."
    )
    A("")

    # Table 1: rank-1 agreement at k=2000 ------------------------------------
    A("## 1. Rank-1 agreement at k=2000")
    A("")
    A("Fraction of queries where the two rerankers' top-1 doc is identical "
      "(pairwise), and where all three agree.")
    A("")
    A("| Dataset | n | C–V | C–Z | V–Z | Three-way |")
    A("|---|---|---|---|---|---|")
    for d in datasets:
        p = all_payloads[d].get(2000)
        if not p:
            continue
        ra = p["rank1_agreement"]
        A(
            f"| {d} | {p['num_queries_intersection']} | "
            f"{_f(ra['cohere_voyage'])} | {_f(ra['cohere_zerank'])} | "
            f"{_f(ra['voyage_zerank'])} | {_f(ra['three_way'])} |"
        )
    A("")

    # Table 2: Venn decomposition at k=2000 ----------------------------------
    A("## 2. R@1 correctness Venn decomposition at k=2000")
    A("")
    A("Fraction of queries by which subset of rerankers placed a gold doc at "
      "rank 1. `headroom` = at least one right and at least one wrong (the "
      "population fusion can act on); `unreachable` = none right (no selector "
      "or fusion over these three can fix rank 1).")
    A("")
    cells = _venn_cells()
    A("| Dataset | " + " | ".join(cells) + " | headroom | unreachable |")
    A("|---|" + "|".join(["---"] * (len(cells) + 2)) + "|")
    for d in datasets:
        p = all_payloads[d].get(2000)
        if not p:
            continue
        fr = p["venn"]["fractions"]
        row = " | ".join(_f(fr[c]) for c in cells)
        A(
            f"| {d} | {row} | {_f(p['venn']['fusion_headroom'])} | "
            f"{_f(p['venn']['unreachable'])} |"
        )
    A("")

    # Table 3: conditional top-1 precision -----------------------------------
    A("## 3. Conditional top-1 precision at k=2000 (agree vs disagree)")
    A("")
    A("`agree` = P(top-1 in gold | all three top-1 identical). `disagree` = "
      "mean per-reranker top-1 precision over queries where they disagree.")
    A("")
    A("| Dataset | agree | n(agree) | disagree | n(disagree) | gap |")
    A("|---|---|---|---|---|---|")
    for d in datasets:
        p = all_payloads[d].get(2000)
        if not p:
            continue
        cp = p["conditional_top1_precision"]
        gap = (
            cp["agree"] - cp["disagree"]
            if cp["agree"] is not None and cp["disagree"] is not None
            else None
        )
        A(
            f"| {d} | {_f(cp['agree'])} | {cp['n_agree']} | "
            f"{_f(cp['disagree'])} | {cp['n_disagree']} | {_f(gap)} |"
        )
    A("")

    # Table 4: oracle-selector + oracle-config + realized fusion at k=2000 ---
    A("## 4. Oracle selector vs oracle-config vs realized fusion (k=2000)")
    A("")
    A("Per metric, side-by-side: hybrid baseline, best singleton, best static "
      "fusion (from existing `runs/`), the **oracle selector** (per-query best "
      "of the three pure singletons), and **oracle-config** (per-query best of "
      "the 29-condition menu of reranked configurations we actually ran).")
    A("")
    A("- `oracle-selector` = max over {(1,0,0), (0,1,0), (0,0,1)} of metric on "
      "the picked singleton's top-K. Rank-respecting per-query oracle over the "
      "three singletons. Upper bound on R-by-routing-singletons.")
    A("- `oracle-config@K` = `max over c in MENU of metric(top_K(ranking_c))` "
      "where MENU is the 29 reranked conditions in `run_experiment.CONDITIONS` "
      "(3 singletons + 26 fusion blends; excludes hybrid_only). Each pick is a "
      "real top-K block — rank-respecting and deployable in principle (train a "
      "classifier to select the config per query). Defined for every metric, "
      "not only recall@K.")
    A("")
    A("By construction, **`oracle-config >= oracle-selector`** (the singletons "
      "are in the menu) and **`oracle-config >= best static fusion`** (a fixed "
      "config is feasible per query). Two decomposition gaps name the prize "
      "for two different next-step strategies:")
    A("")
    A("- **blending value** = `oracle-config − oracle-selector`. How much of "
      "the per-query oracle requires INTERIOR blends rather than pure "
      "per-query model selection (simplex corners). Equality (≈0) means routing "
      "to the right singleton per query is enough.")
    A("- **adaptive-fusion prize** = `oracle-config − best-static-fusion`. What "
      "query-adaptive weight prediction could unlock over the single best "
      "static blend.")
    A("")
    A("Why oracle-config and not oracle-merge: a set-union ceiling credits a "
      "gold doc the moment it appears anywhere in the union of three top-K "
      "lists. No weighted fusion can keep one reranker's gold doc without "
      "inheriting whatever that same weighting promotes above it — so the union "
      "credit overstates what any fusion can realize. Oracle-config replaces "
      "that loose ceiling with a per-query max over the rank-respecting top-K "
      "outputs of real menu conditions. (Prior reports' oracle-merge / "
      "pooled-union columns and their \"combining headroom\" framing are "
      "discontinued; both were artifacts of free subset selection.)")
    A("")
    metric_titles = {
        "recall_at_1": "R@1",
        "recall_at_5": "R@5",
        "recall_at_20": "R@20",
        "nDCG_at_10": "nDCG@10",
    }
    for d in datasets:
        p = all_payloads[d].get(2000)
        if not p:
            continue
        comp = p["comparison_from_runs"]
        A(f"### {d} (n={p['num_queries_intersection']})")
        A("")
        A("| Metric | hybrid | best singleton | best fusion | oracle-selector | "
          "oracle-config | blending value | adaptive-fusion prize |")
        A("|---|---|---|---|---|---|---|---|")
        for mkey, mt in metric_titles.items():
            hybrid = comp.get("hybrid_only", {}).get(mkey) if comp.get("hybrid_only") else None
            bs = comp.get("best_singleton", {}).get(mkey)
            bf = comp.get("best_fusion", {}).get(mkey)
            os_ = p["oracle_selector"][f"avg_{mkey}"]
            oc = p["oracle_config"][f"avg_{mkey}"]
            blending = oc - os_
            bf_val = bf["value"] if bf else None
            adaptive = (oc - bf_val) if bf_val is not None else None
            A(
                f"| {mt} | {_f(hybrid)} | {_cell(bs)} | {_cell(bf)} | "
                f"{_f(os_)} | {_f(oc)} | {_f(blending)} | {_f(adaptive)} |"
            )
        A("")

    # Appendix: oracle at other k --------------------------------------------
    A("## 5. Rank-1 agreement vs depth (per dataset, all k)")
    A("")
    A("Pairwise rank-1 agreement at each retrieved_k. The hypothesis: Cohere's "
      "agreement with Voyage and Zerank decays as k grows on Economics and "
      "Psychology (the Cohere-regression domains), while Voyage–Zerank stays "
      "flat.")
    A("")
    for d in datasets:
        A(f"### {d}")
        A("")
        A("| k | C–V | C–Z | V–Z | three-way | mean Kendall τ |")
        A("|---|---|---|---|---|---|")
        for k in K_VALUES:
            p = all_payloads[d].get(k)
            if not p:
                continue
            ra = p["rank1_agreement"]
            tau = p["kendall_tau"]
            tau_mean = sum(tau.values()) / len(tau)
            A(
                f"| {k} | {_f(ra['cohere_voyage'])} | {_f(ra['cohere_zerank'])} | "
                f"{_f(ra['voyage_zerank'])} | {_f(ra['three_way'])} | "
                f"{_f(tau_mean)} |"
            )
        A("")

    # Appendix: oracle gap at all k ------------------------------------------
    A("## 6. Oracle gap appendix (oracle selector − best static fusion)")
    A("")
    A("Oracle R@1 / R@20 minus the best static fusion at the same cell — the "
      "quantified prize for query-adaptive routing.")
    A("")
    A("| Dataset | k | oracle R@1 | best fusion R@1 | gap R@1 | oracle R@20 | best fusion R@20 | gap R@20 |")
    A("|---|---|---|---|---|---|---|---|")
    for d in datasets:
        for k in K_VALUES:
            p = all_payloads[d].get(k)
            if not p:
                continue
            comp = p["comparison_from_runs"]
            bf1 = comp.get("best_fusion", {}).get("recall_at_1")
            bf20 = comp.get("best_fusion", {}).get("recall_at_20")
            o1 = p["oracle_selector"]["avg_recall_at_1"]
            o20 = p["oracle_selector"]["avg_recall_at_20"]
            g1 = o1 - bf1["value"] if bf1 else None
            g20 = o20 - bf20["value"] if bf20 else None
            A(
                f"| {d} | {k} | {_f(o1)} | {_f(bf1['value'] if bf1 else None)} | "
                f"{_f(g1)} | {_f(o20)} | {_f(bf20['value'] if bf20 else None)} | "
                f"{_f(g20)} |"
            )
    A("")

    out_path = RESULTS_DIR / "AGREEMENT.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS.keys()), default=None)
    parser.add_argument("--retrieved-k", type=int, default=None,
                        help="Single k (default: all five).")
    parser.add_argument("--all-datasets", action="store_true",
                        help="Run every completed dataset x all k, then write AGREEMENT.md.")
    parser.add_argument("--simplex-grid", action="store_true",
                        help=("Optional second pass: per-query 3-simplex grid sweep over RSF "
                              "and RRF weight vectors (231 points at step 0.05) using "
                              "fuse_rsf / fuse_rrf. LOWER bound on the "
                              "continuous fusion-family optimum. Off by default — adds compute, "
                              "still zero-API."))
    parser.add_argument("--smoke", action="store_true",
                        help="First 10 queries, biology, k=100 only; run invariant checks.")
    args = parser.parse_args()

    if args.smoke:
        run_smoke()
        return

    if args.all_datasets:
        all_payloads: dict[str, dict[int, dict]] = {}
        for d in DATASETS:
            print(f"\n=== {d} ===")
            loaded = load_and_validate(d)
            if loaded is None:
                continue
            cache, qs = loaded
            all_payloads[d] = {}
            for k in K_VALUES:
                payload = analyze_dataset_k(d, cache, qs, k, simplex_grid=args.simplex_grid)
                all_payloads[d][k] = payload
                if not payload["invariants_ok"]:
                    print(f"  [WARN] invariant errors at k={k}: {payload['invariant_errors']}")
        write_agreement_md(all_payloads)
        return

    if not args.dataset:
        parser.error("provide --dataset, --all-datasets, or --smoke")
    loaded = load_and_validate(args.dataset)
    if loaded is None:
        raise SystemExit(f"No usable cache for {args.dataset}.")
    cache, qs = loaded
    ks = [args.retrieved_k] if args.retrieved_k else list(K_VALUES)
    for k in ks:
        print(f"\n--- {args.dataset} k={k} ---")
        payload = analyze_dataset_k(args.dataset, cache, qs, k, simplex_grid=args.simplex_grid)
        if not payload["invariants_ok"]:
            print(f"  [WARN] invariant errors: {payload['invariant_errors']}")


def run_smoke() -> None:
    """Smoke test: biology, k=100, first 10 intersection queries; assert invariants."""
    print("=== SMOKE: biology k=100, first 10 queries ===")
    dataset_slug = "biology"
    loaded = load_and_validate(dataset_slug)
    if loaded is None:
        raise SystemExit("Smoke needs the biology k=2000 cache.")
    cache, qs = loaded

    # Restrict the query set to the first 10 intersection queries.
    keep = list(qs.gold.keys())[:10]
    qs.gold = {t: qs.gold[t] for t in keep}
    qs.query_ids = {t: qs.query_ids.get(t, t[:64]) for t in keep}
    for p in PROVIDERS:
        qs.present_by_provider[p] = {
            t: g for t, g in qs.present_by_provider[p].items() if t in keep
        }

    payload = analyze_dataset_k(
        dataset_slug, cache, qs, 100, write=False, read_prior=False,
    )

    # Hard invariant asserts (spec 9.4).
    assert payload["invariants_ok"], payload["invariant_errors"]
    n = payload["num_queries_intersection"]
    assert n == len(keep), f"expected {len(keep)} queries, got {n}"
    # Jaccard of a list with itself is 1.0 — sanity-check via a manual pair.
    text = keep[0]
    rk = _full_rankings(cache, text, 100, list(PROVIDERS))
    for r in PROVIDERS:
        assert _jaccard(rk[r][:20], rk[r][:20]) == 1.0
    # Venn counts sum to n.
    assert sum(payload["venn"]["counts"].values()) == n
    # Oracle-config >= oracle-selector >= singletons (echo of invariants).
    for m in CAP20_METRICS:
        os_ = payload["oracle_selector"][f"avg_{m}"]
        oc = payload["oracle_config"][f"avg_{m}"]
        assert oc >= os_ - 1e-9, f"oracle_config {m} {oc} < oracle_selector {m} {os_}"
    # Oracle-config >= best static fusion (where comparison is available).
    comp = payload.get("comparison_from_runs", {}) or {}
    bf = comp.get("best_fusion") or {}
    for m in CAP20_METRICS:
        rec = bf.get(m)
        if rec:
            oc = payload["oracle_config"][f"avg_{m}"]
            assert oc >= rec["value"] - 1e-9, (
                f"oracle_config {m} {oc} < best static fusion {rec['condition']} {rec['value']}"
            )
    # Oracle-selector R@1 == 1 - venn[none].
    assert abs(payload["oracle_selector"]["avg_recall_at_1"]
               - (1.0 - payload["venn"]["fractions"]["none"])) < 1e-9

    print("\nSMOKE PASSED. Key numbers:")
    print(f"  n={n}")
    print(f"  rank1 agreement: {payload['rank1_agreement']}")
    print(f"  venn fractions: {payload['venn']['fractions']}")
    print(f"  oracle_selector: {payload['oracle_selector']}")
    print(f"  oracle_config:   {payload['oracle_config']}")
    if "oracle_config_cap100" in payload:
        print(f"  oracle_config_cap100: {payload['oracle_config_cap100']}")
    print(f"  conditional precision: {payload['conditional_top1_precision']}")


if __name__ == "__main__":
    main()
