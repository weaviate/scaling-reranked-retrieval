"""Measure across-trial variance of first-stage hybrid retrieval recall.

Why: the MoCE evaluation runs each condition once (num_trials=1). The rerankers
are justified single-trial (Voyage/Zerank bit-identical, Cohere negligible float
noise — see score_variance.py). This closes the remaining gap: is the *retrieval*
itself stable enough across runs that a single hybrid pass is honest?

We answer it by re-running the exact `hybrid_only` condition (Weaviate hybrid,
BM25 + Arctic 2.0, relative-score fusion — the same BaseRetriever the main
experiment uses) `n_trials` times per subset at retrieved_k=2000, and looking at
the spread of mean recall@k across trials. Because qab's orchestrator re-invokes
the agent on every trial, each trial is a genuinely fresh query-time retrieval —
exactly the variation a deployed system sees. The index is held fixed (no
re-ingest), so the only varying input is the query.

This reuses the *same* code path as the paper (run_search_eval + the hybrid_only
BaseRetriever + the same recall cutoffs), so the means it reports are the very
numbers Section 4.1 plots — not a re-implementation. qab already does the
per-trial retrieval, the per-query mean, and across-trial mean/std/min/max
(analysis.aggregate_metrics); this script only reshapes that into a methodology
table, a machine-readable summary, and optional error-bar inputs.

Two metric families are reported:
    pool_recall : mean recall@k for k in {100,200,500,1000,2000} — presence in
                  the top-k candidate pool. This is the ceiling the rerankers
                  inherit (what 4.1 plots). Near-saturated, so low variance by
                  construction.
    ranked      : mean recall@1 and nDCG@10 on the hybrid ranking itself, before
                  any rerank. Rank-1 identity is far more sensitive to fusion
                  tie-flips than pool membership (score_variance.py has already
                  observed a fresh hybrid top-1 differ from a cached one), so
                  this is the stricter test — the one a reviewer attacks.

Does NOT touch the downstream rerank score caches (results/.../caches/): it only
reads gold sets via run_search_eval and writes to results/.../extras/ +
results/hybrid_bounds_*.

Two views:
    single-k (default) : retrieve once at retrieved_k=2000 and read variance at
                  every cutoff. This is the right denominator for the MAIN
                  experiment's reported numbers, which all derive from one k=2000
                  cache. Writes results/hybrid_bounds_{summary.json,table.md,
                  errorbars.json}; --rebuild re-renders the table offline.
    --k-sweep   : fresh-retrieve at EACH retrieved_k in {100,200,500,1000,2000}
                  (independent retrievals, not one pool truncated) and report
                  R@1/R@20 across-trial variance vs retrieved_k. A `limit=k`
                  hybrid call sets HNSW's dynamic ef AND the relativeScoreFusion
                  normalization pool, so smaller k is a different (and possibly
                  noisier) operation — this view bounds the k=100 operating point
                  and exposes the direct-vs-derived gap the cache shortcut hides.
                  Writes results/hybrid_bounds_vs_k.{json,md}.

Usage:
    export WEAVIATE_URL=... WEAVIATE_API_KEY=...

    uv run python scripts/hybrid_variance.py
    uv run python scripts/hybrid_variance.py --k-sweep
    uv run python scripts/hybrid_variance.py --k-sweep --datasets biology --num-samples 10
    uv run python scripts/hybrid_variance.py --rebuild   # offline re-render
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from query_agent_benchmarking import run_search_eval

from src.adapters.retrieval.base_retriever import BaseRetriever

from src.adapters.qab import RetrieverSearchAgent
from src.config import (
    DATASETS,
    RANDOM_SEED,
    RESULTS_DIR,
    get_results_dir,
)
from src.domain.metrics import build_extra_metrics

from src.adapters import qab

# qab>=0.7 guard + load_search_dataset memoization (keeps this in lockstep
# with the main experiment).
qab.setup()

# BRIGHT subsets, in the paper's order. IRPAPERS is excluded — it isn't part of
# the Section 4.1 hybrid scaling chart.
DEFAULT_SUBSETS = ["biology", "earth_science", "economics", "psychology", "robotics"]

# Candidate-pool recall cutoffs (the Section 4.1 scaling-chart x-axis). Reported
# as a secondary family — these are the ceiling the rerankers inherit, but they
# are near-saturated and multi-gold, so a recall delta there isn't one query.
POOL_RECALL_KS = [100, 200, 500, 1000, 2000]

# HEADLINE ranked metrics, in the experiment's stated priority order (CLAUDE.md
# "Metric priorities"): R@1 ("give me the answer") and R@20 ("give me a candidate
# list") are primary; R@5 is the mid-range sidekick; nDCG@10 the ordering one.
# All four are computed on the hybrid order itself, before any rerank, and are
# already in qab's BRIGHT default metric profile (recall@1/5/20 + nDCG@10), so
# surfacing R@5/R@20 here costs no extra retrieval — only a re-run to repopulate.
# Maps a display name -> the qab metric key found in each per-trial dict.
RANKED_METRICS = {
    "recall@1": "avg_recall_at_1",
    "recall@5": "avg_recall_at_5",
    "recall@20": "avg_recall_at_20",
    "nDCG@10": "avg_nDCG_at_10",
}

# Which ranked metrics are the experiment's primary headlines (drive the verdict
# and the movers callout); the rest are sidekicks shown for completeness.
HEADLINE_RANKED = ("recall@1", "recall@20")

# Smallest lift the paper treats as real, PER METRIC, so the variance verdict
# divides each metric's own across-trial std by an effect measured in the SAME
# units (the old table wrongly divided the R@1 effect by the pool-recall std and
# printed a meaningless "1×"). R@1/R@5 single-query ≈ 0.010; smallest reported
# R@1 fusion lift is +0.010, R@20 noise floor is ~0.010 (see CLAUDE.md Caveats),
# nDCG@10's flat fusion lift is ~+0.012.
SMALLEST_EFFECT_BY_METRIC = {
    "recall@1": 0.010,
    "recall@5": 0.010,
    "recall@20": 0.010,
    "nDCG@10": 0.012,
}
# Back-compat alias (R@1 effect) for any external reader of the summary JSON.
SMALLEST_REPORTED_EFFECT = SMALLEST_EFFECT_BY_METRIC["recall@1"]

# Full-query-set sizes per BRIGHT subset (num_samples=None runs the whole set).
# These convert an across-trial recall *range* into "gold-doc boundary crossings"
# = range x Q: mean recall@k only moves when a gold doc crosses the rank-k line,
# so range x Q is the integer number of single-gold-query-equivalents that
# flipped between trials — the unit a reviewer can actually reason about.
# Verified exact against the data: earth_science R@1 range 0.008621 == 1/116.
SUBSET_QUERY_COUNTS = {
    "biology": 103,
    "earth_science": 116,
    "economics": 103,
    "psychology": 101,
    "robotics": 101,
}

DEFAULT_RETRIEVED_K = 2000

# --- retrieved_k sweep (the operating-point variance view) ---------------------
# The main experiment derives every smaller-k result from one k=2000 collection,
# so the k=2000 variance is the right denominator for THOSE numbers. But a real
# `retrieved_k=k` retrieval is a *different operation*: the hybrid call issues
# `collection.query.hybrid(limit=k)` with no ef/fusion overrides, so (a) HNSW's
# dynamic ef scales with the limit (smaller k => narrower, more approximate
# search) and (b) relativeScoreFusion min-max normalizes over the returned pool
# of size k (so the fused HEAD order is pool-size sensitive). The sweep
# fresh-retrieves at each k and measures across-trial variance per operating
# point — leading with k=100 as the conservative bound — and exposes the
# direct-vs-derived gap (mean at k vs mean at the largest k = the derived value).
DEFAULT_SWEEP_KS = [100, 200, 500, 1000, 2000]

# Headline cutoffs the sweep matrix tracks (both <= every swept k, so always
# measurable). These are the two metrics the experiment centers on.
SWEEP_METRICS = {
    "recall@1": "avg_recall_at_1",
    "recall@20": "avg_recall_at_20",
}


def _key_for_k(k: int) -> str:
    return f"avg_recall_at_{k}"


def _stats(values: list[float], q: int | None = None) -> dict:
    """Across-trial summary for one (subset, metric) cell.

    std is the *sample* std (ddof=1), matching the spec — qab's own aggregate
    uses population std (ddof=0); we recompute from the raw per-trial values so
    the convention is explicit. With <2 trials std/range collapse to 0.

    If `q` (the subset's query count) is given, also report `gold_flips` =
    range x q — the integer-valued number of single-gold-query boundary
    crossings that the across-trial range represents (mean recall@k moves only
    when a gold doc crosses the rank-k line). This is the interpretable unit:
    "1.0 flips" == "one query changed its hit/miss between trials".
    """
    n = len(values)
    mean = statistics.fmean(values) if values else 0.0
    std = statistics.stdev(values) if n > 1 else 0.0
    lo = min(values) if values else 0.0
    hi = max(values) if values else 0.0
    out = {
        "mean": mean,
        "std": std,
        "min": lo,
        "max": hi,
        "range": hi - lo,
        "n": n,
    }
    if q:
        out["q"] = q
        out["gold_flips"] = (hi - lo) * q
    return out


def collect_per_trial(agg: dict) -> dict[str, list[float]]:
    """Pull raw per-trial values for every metric out of qab's aggregate.

    Returns {metric_key: [v_trial0, v_trial1, ...]} preserving trial order.
    """
    trials = agg.get("trials", [])
    if not trials:
        return {}
    keys = [k for k in trials[0] if k.startswith("avg_")]
    return {key: [t[key] for t in trials if key in t] for key in keys}


def run_subset(
    subset: str,
    n_trials: int,
    retrieved_k: int,
    num_samples: int | None,
    max_concurrent: int,
    use_async: bool,
) -> dict[str, list[float]]:
    """Run the hybrid_only condition n_trials times; return per-trial metrics."""
    cfg = DATASETS[subset]
    retriever = BaseRetriever(
        collection_name=cfg.collection,
        target_property_name=cfg.target_property,
        retrieved_k=retrieved_k,
        search_type="hybrid",
        verbose=False,
    )
    agent = RetrieverSearchAgent(retriever)
    extra_metrics = build_extra_metrics(retrieved_k, cfg)

    # Raw qab aggregate is parked in extras/ (never caches/) for provenance.
    out_path = get_results_dir(subset) / "extras" / f"hybrid_variance_k{retrieved_k}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {subset} ({cfg.qab_name}) — {n_trials} trials @ k={retrieved_k} ===")
    agg = run_search_eval(
        search_dataset=cfg.qab_name,
        search_agent=agent,
        agent_name="hybrid_only",
        use_async=use_async,
        num_trials=n_trials,
        use_subset=num_samples is not None,
        num_samples=num_samples,
        random_seed=RANDOM_SEED,
        output_path=str(out_path),
        extra_metrics=extra_metrics,
        max_concurrent=max_concurrent,
    )
    per_trial = collect_per_trial(agg)
    print(f"  collected {len(per_trial)} metrics over {agg.get('num_trials')} trials")
    return per_trial


def median_across_subsets(
    per_subset_trials: dict[str, dict[str, list[float]]],
    metric_key: str,
) -> dict:
    """Across-trial stats of the median-over-subsets line for one metric.

    For each trial t, take the median over subsets of that subset's value at
    trial t (the paper's "Median" series is a median across the five subsets),
    then summarize that per-trial median series across trials.
    """
    series = [v for v in per_subset_trials.values() if metric_key in v]
    if not series:
        return _stats([])
    n_trials = min(len(v[metric_key]) for v in series)
    per_trial_median = [
        statistics.median(v[metric_key][t] for v in series) for t in range(n_trials)
    ]
    return _stats(per_trial_median)


def _argmax_cell(per_subset: dict[str, dict[str, dict]], field: str) -> dict:
    """Find the (subset, cell) with the largest `field`, for the headline."""
    best = {"value": 0.0, "subset": None, "cell": None}
    for subset, cells in per_subset.items():
        for cell, stats in cells.items():
            if stats[field] > best["value"]:
                best = {"value": stats[field], "subset": subset, "cell": cell}
    return best


def _q_for(subset: str) -> int | None:
    return SUBSET_QUERY_COUNTS.get(subset)


def _per_metric_verdict(ranked_per_subset: dict) -> dict:
    """For each ranked metric, the worst across-trial std/range over subsets and
    the margin against THAT metric's own smallest reported effect.

    Fixes the old single-denominator bug: R@1 noise is judged against the R@1
    effect, R@20 noise against the R@20 effect, etc. — never against pool recall.
    """
    out: dict[str, dict] = {}
    for name in RANKED_METRICS:
        worst_std = {"value": 0.0, "subset": None}
        worst_flips = {"value": 0.0, "subset": None}
        for subset, cells in ranked_per_subset.items():
            c = cells.get(name)
            if not c:
                continue
            if c["std"] > worst_std["value"]:
                worst_std = {"value": c["std"], "subset": subset}
            if c.get("gold_flips", 0.0) > worst_flips["value"]:
                worst_flips = {"value": c["gold_flips"], "subset": subset}
        effect = SMALLEST_EFFECT_BY_METRIC.get(name)
        margin = (effect / worst_std["value"]) if (effect and worst_std["value"] > 0) else None
        out[name] = {
            "max_std": worst_std,
            "max_gold_flips": worst_flips,
            "smallest_effect_same_metric": effect,
            "effect_over_max_std": margin,
            "bit_identical_everywhere": worst_std["value"] == 0.0,
            "is_headline": name in HEADLINE_RANKED,
        }
    return out


def build_summary(
    per_subset_trials: dict[str, dict[str, list[float]]],
    config: dict,
) -> dict:
    # --- pool recall family ---
    pool_per_subset: dict[str, dict[str, dict]] = {}
    for subset, trials in per_subset_trials.items():
        q = _q_for(subset)
        pool_per_subset[subset] = {
            str(k): _stats(trials.get(_key_for_k(k), []), q=q)
            for k in POOL_RECALL_KS
            if _key_for_k(k) in trials
        }
    pool_median = {
        str(k): median_across_subsets(per_subset_trials, _key_for_k(k))
        for k in POOL_RECALL_KS
    }

    # --- ranked family ---
    ranked_per_subset: dict[str, dict[str, dict]] = {}
    for subset, trials in per_subset_trials.items():
        q = _q_for(subset)
        ranked_per_subset[subset] = {
            name: _stats(trials.get(key, []), q=q)
            for name, key in RANKED_METRICS.items()
            if key in trials
        }
    ranked_median = {
        name: median_across_subsets(per_subset_trials, key)
        for name, key in RANKED_METRICS.items()
    }

    def headline(per_subset: dict) -> dict:
        std = _argmax_cell(per_subset, "std")
        rng = _argmax_cell(per_subset, "range")
        ratio = (SMALLEST_REPORTED_EFFECT / std["value"]) if std["value"] > 0 else None
        return {
            "max_std_observed": std,
            "max_range_observed": rng,
            "smallest_reported_effect": SMALLEST_REPORTED_EFFECT,
            "effect_over_max_std_ratio": ratio,
        }

    return {
        "config": config,
        "pool_recall": {
            "per_subset": pool_per_subset,
            "median_across_subsets": pool_median,
            **headline(pool_per_subset),
        },
        "ranked": {
            "per_subset": ranked_per_subset,
            "median_across_subsets": ranked_median,
            "per_metric_verdict": _per_metric_verdict(ranked_per_subset),
            **headline(ranked_per_subset),
        },
    }


def _spread_cell(c: dict | None) -> str:
    """One ranked cell as `mean` plus its across-trial spread in QUERY units.

    Zero-variance cells (the common case) read `0.500  (0)` so the eye finds the
    movers; movers read `0.209  (±.0039, 1.0q)`.
    """
    if not c:
        return "n/a"
    if c["std"] == 0.0:
        return f"{c['mean']:.3f}  (0)"
    flips = c.get("gold_flips")
    flip_txt = f", {flips:.1f}q" if flips is not None else ""
    return f"{c['mean']:.3f}  (±{c['std']:.4f}{flip_txt})"


def build_table(summary: dict) -> str:
    cfg = summary["config"]
    lines: list[str] = []
    lines.append("# Hybrid Search Recall Variance — Across-Trial Bounds\n")
    alpha_note = cfg.get("alpha_note", "")
    lines.append(
        f"First-stage hybrid retrieval (BM25 + Arctic 2.0, "
        f"{cfg['fusion']} fusion, alpha={cfg['alpha']} — {alpha_note}) re-run "
        f"**{cfg['n_trials']} times** per subset at retrieved_k={cfg['retrieved_k']}, "
        f"index held fixed, no client-side seed. Each trial is a genuinely fresh "
        f"query-time retrieval.\n"
    )
    lines.append(
        "**Reading the spread.** Mean recall@k changes across trials only when a "
        "gold doc crosses the rank-k boundary, so we report the spread in "
        "*queries*: `Nq` = (max−min across the 5 trials) × Q = the number of "
        "single-gold-query flips. `(0)` = bit-identical across all 5 trials. "
        "Cells are `mean  (±std, Nq)`.\n"
    )

    # --- 1. The verdict, up top, per headline metric (R@1 / R@20) ---
    lines.append("## Verdict — do R@1 and R@20 move across trials?\n")
    verdict = summary["ranked"].get("per_metric_verdict", {})
    for name in RANKED_METRICS:
        v = verdict.get(name)
        if not v:
            continue
        tag = "**primary**" if v.get("is_headline") else "sidekick"
        if v["bit_identical_everywhere"]:
            lines.append(
                f"- **{name}** ({tag}): **bit-identical across all 5 trials in "
                f"every subset** — zero across-trial movement.\n"
            )
        else:
            s = v["max_std"]
            f = v["max_gold_flips"]
            margin = v["effect_over_max_std"]
            eff = v["smallest_effect_same_metric"]
            margin_txt = (
                f"**{margin:.1f}× below** the smallest reported {name} effect (+{eff:.3f})"
                if margin is not None else "n/a"
            )
            lines.append(
                f"- **{name}** ({tag}): only **{s['subset']}** moves — worst "
                f"across-trial std **{s['value']:.4f}** "
                f"(≈ {f['value']:.1f} query at {f['subset']}); all other subsets "
                f"bit-identical. That std is {margin_txt}.\n"
            )

    # --- 2. Headline ranked table (R@1 / R@5 / R@20 / nDCG@10) ---
    lines.append("## Headline cutoffs (hybrid order, pre-rerank)\n")
    names = list(RANKED_METRICS.keys())
    header = [n + (" ★" if n in HEADLINE_RANKED else "") for n in names]
    lines.append("| Subset | Q | " + " | ".join(header) + " |")
    lines.append("|" + "---|" * (len(names) + 2))
    for subset, cells in summary["ranked"]["per_subset"].items():
        q = SUBSET_QUERY_COUNTS.get(subset, "?")
        row = [subset, str(q)] + [_spread_cell(cells.get(n)) for n in names]
        lines.append("| " + " | ".join(row) + " |")
    medr = summary["ranked"]["median_across_subsets"]
    row = ["**Median**", "—"] + [
        (f"**{medr[n]['mean']:.3f}**" if n in medr else "n/a") for n in names
    ]
    lines.append("| " + " | ".join(row) + " |")
    lines.append("\n★ = primary metric (CLAUDE.md priority). Median is across "
                 "subsets; its spread is omitted (Q differs by subset).\n")

    # --- 3. Movers-only callout ---
    lines.append("## Movers only (every non-bit-identical cell)\n")
    movers: list[str] = []
    for family, block in (("R@k", summary["ranked"]), ("poolR@k", summary["pool_recall"])):
        for subset, cells in block["per_subset"].items():
            for cell, c in cells.items():
                if c["std"] > 0.0:
                    flips = c.get("gold_flips")
                    ftxt = f", {flips:.1f} flips" if flips is not None else ""
                    label = cell if family == "R@k" else f"R@{cell}"
                    movers.append(
                        f"- {subset} / {label}: range {c['range']:.4f} "
                        f"(±{c['std']:.4f}{ftxt})"
                    )
    if movers:
        lines.extend(movers)
        lines.append("")
    else:
        lines.append("- (none — every cell bit-identical across all 5 trials)\n")

    # --- 4. Pool recall family (the §4.1 ceiling), kept for completeness ---
    lines.append("## Pool recall@k (candidate-set ceiling — secondary)\n")
    lines.append(
        "Near-saturated and multi-gold, so a recall delta here is several "
        "boundary crossings, not one query. Reported for the §4.1 scaling chart.\n"
    )
    ks = [str(k) for k in POOL_RECALL_KS]
    lines.append("| Subset | " + " | ".join(f"R@{k}" for k in ks) + " |")
    lines.append("|" + "---|" * (len(ks) + 1))
    for subset, cells in summary["pool_recall"]["per_subset"].items():
        row = [subset] + [_spread_cell(cells.get(k)) for k in ks]
        lines.append("| " + " | ".join(row) + " |")
    med = summary["pool_recall"]["median_across_subsets"]
    row = ["**Median**"] + [
        (f"**{med[k]['mean']:.3f}**" if k in med else "n/a") for k in ks
    ]
    lines.append("| " + " | ".join(row) + " |\n")

    return "\n".join(lines)


def build_errorbars(summary: dict) -> dict:
    """Per-(subset, k) mean+std for the pool-recall family, shaped for plotting.

    One series per subset plus a Median series, each a parallel {k, mean, std}.
    NOTE: confirm this matches the Section 4.1 plotting code's expected shape;
    if it consumes a different layout, reshape here rather than in the plotter.
    """
    out: dict[str, dict] = {}
    ks = [str(k) for k in POOL_RECALL_KS]
    for subset, cells in summary["pool_recall"]["per_subset"].items():
        present = [k for k in ks if k in cells]
        out[subset] = {
            "k": [int(k) for k in present],
            "mean": [cells[k]["mean"] for k in present],
            "std": [cells[k]["std"] for k in present],
        }
    med = summary["pool_recall"]["median_across_subsets"]
    present = [k for k in ks if k in med]
    out["Median"] = {
        "k": [int(k) for k in present],
        "mean": [med[k]["mean"] for k in present],
        "std": [med[k]["std"] for k in present],
    }
    return out


def run_k_sweep(
    subsets: list[str],
    sweep_ks: list[int],
    n_trials: int,
    num_samples: int | None,
    max_concurrent: int,
    use_async: bool,
) -> dict[str, dict[int, dict[str, list[float]]]]:
    """Fresh-retrieve n_trials times at EACH retrieved_k, per subset.

    Returns {subset: {k: {metric_key: [per-trial values]}}}. Each (subset, k) is
    a genuinely separate hybrid retrieval at limit=k (not a truncation of a
    larger pool), so its across-trial spread reflects that operating point's own
    HNSW-ef + relativeScoreFusion-pool behavior.
    """
    out: dict[str, dict[int, dict[str, list[float]]]] = {}
    for subset in subsets:
        out[subset] = {}
        for k in sweep_ks:
            try:
                out[subset][k] = run_subset(
                    subset,
                    n_trials=n_trials,
                    retrieved_k=k,
                    num_samples=num_samples,
                    max_concurrent=max_concurrent,
                    use_async=use_async,
                )
            except Exception as e:  # noqa: BLE001
                print(f"  FAILED {subset} k={k}: {e!r}")
                out[subset][k] = {}
    return out


def build_sweep_summary(
    sweep_trials: dict[str, dict[int, dict[str, list[float]]]],
    sweep_ks: list[int],
    config: dict,
) -> dict:
    """Per-metric {subset: {k: across-trial stats}} for the R@1 / R@20 matrix."""
    metrics: dict[str, dict] = {}
    for mname, mkey in SWEEP_METRICS.items():
        per_subset: dict[str, dict[str, dict]] = {}
        for subset, byk in sweep_trials.items():
            q = _q_for(subset)
            per_subset[subset] = {
                str(k): _stats(byk.get(k, {}).get(mkey, []), q=q)
                for k in sweep_ks
                if byk.get(k, {}).get(mkey)
            }
        metrics[mname] = {"per_subset": per_subset}
    return {"config": config, "sweep_ks": sweep_ks, "metrics": metrics}


def build_sweep_table(summary: dict) -> str:
    cfg = summary["config"]
    sweep_ks = summary["sweep_ks"]
    kmax = max(sweep_ks)
    lines: list[str] = []
    lines.append("# Hybrid Search Recall Variance — Across-Trial Bounds vs. retrieved_k\n")
    lines.append(
        f"Each (subset, k) is a **fresh** hybrid retrieval at `limit=k` "
        f"(BM25 + Arctic 2.0, {cfg['fusion']} fusion, alpha={cfg['alpha']}), re-run "
        f"**{cfg['n_trials']} times**, index fixed, no client-side seed. Unlike the "
        f"main experiment (which derives every k from one k={kmax} cache), these are "
        f"independent retrievals — so the spread at each column is that operating "
        f"point's OWN variance.\n"
    )
    lines.append(
        "**Why k matters.** `limit=k` sets HNSW's dynamic ef (smaller k => narrower, "
        "more approximate search) and the relativeScoreFusion normalization pool "
        "(fused head order is pool-size sensitive). So the k=" + str(kmax) + " column "
        "is the most-exhaustive case and the value the main experiment actually uses; "
        "the k=100 column is the conservative operating-point bound.\n"
    )
    lines.append(
        "Cells: `mean  (±std, Nq)` where `Nq` = (max−min across trials) × Q = "
        "single-gold-query flips; `(0)` = bit-identical across all trials.\n"
    )

    for mname in SWEEP_METRICS:
        block = summary["metrics"].get(mname, {}).get("per_subset", {})
        if not block:
            continue
        star = " ★" if mname in HEADLINE_RANKED else ""
        lines.append(f"## {mname}{star} — across-trial spread by retrieved_k\n")
        kcols = [str(k) for k in sweep_ks]
        lines.append("| Subset | Q | " + " | ".join(f"k={k}" for k in kcols) + " |")
        lines.append("|" + "---|" * (len(kcols) + 2))
        for subset, cells in block.items():
            q = SUBSET_QUERY_COUNTS.get(subset, "?")
            row = [subset, str(q)] + [_spread_cell(cells.get(k)) for k in kcols]
            lines.append("| " + " | ".join(row) + " |")

        # Footer 1 — does stability change with k? worst across-trial std per k.
        worst_std = []
        for k in kcols:
            stds = [cells[k]["std"] for cells in block.values() if k in cells]
            worst_std.append(max(stds) if stds else 0.0)
        lines.append(
            "| **worst std (any subset)** | — | "
            + " | ".join(f"{s:.4f}" for s in worst_std) + " |"
        )

        # Footer 2 — direct-vs-derived: worst |mean(k) − mean(kmax)| across subsets.
        worst_drift = []
        for k in kcols:
            drifts = []
            for cells in block.values():
                if k in cells and str(kmax) in cells:
                    drifts.append(abs(cells[k]["mean"] - cells[str(kmax)]["mean"]))
            worst_drift.append(max(drifts) if drifts else 0.0)
        lines.append(
            f"| **worst Δmean vs k={kmax}** | — | "
            + " | ".join(f"{d:.4f}" for d in worst_drift) + " |"
        )
        lines.append("")

        # One-line verdict for this metric.
        trend = (
            "rises with k" if worst_std[-1] > worst_std[0] + 1e-9
            else "falls with k" if worst_std[-1] + 1e-9 < worst_std[0]
            else "flat across k"
        )
        max_drift = max(worst_drift) if worst_drift else 0.0
        lines.append(
            f"**{mname} reading.** Worst across-trial std {trend} "
            f"(k=100: {worst_std[0]:.4f} -> k={kmax}: {worst_std[-1]:.4f}). "
            f"Largest direct-vs-derived mean gap (any subset, any k vs k={kmax}) "
            f"= {max_drift:.4f} = {max_drift * 100:.1f} pts — the relativeScoreFusion "
            f"+ ef pool-size sensitivity the derive-from-k{kmax} shortcut hides.\n"
        )

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_SUBSETS,
        choices=sorted(DATASETS.keys()),
        help="BRIGHT subsets to measure (default: all five).",
    )
    p.add_argument("--n-trials", type=int, default=5,
                   help="Number of independent retrieval trials per subset.")
    p.add_argument("--retrieved-k", type=int, default=DEFAULT_RETRIEVED_K,
                   help="Pool size retrieved once per trial; smaller k are prefixes.")
    p.add_argument("--num-samples", type=int, default=None,
                   help="Limit queries per subset (smoke testing).")
    p.add_argument("--max-concurrent", type=int, default=8,
                   help="Concurrent queries inside qab. No reranker TPM limit "
                        "applies here, so this can be higher than the main sweep.")
    p.add_argument("--sync", action="store_true", help="Sync execution.")
    p.add_argument(
        "--k-sweep", action="store_true",
        help="Operating-point mode: fresh-retrieve n_trials times at EACH "
             "--sweep-ks value (not derived from one pool) and report R@1/R@20 "
             "across-trial variance vs retrieved_k + the direct-vs-derived gap. "
             "Writes hybrid_bounds_vs_k.{md,json}.",
    )
    p.add_argument(
        "--sweep-ks", type=int, nargs="+", default=DEFAULT_SWEEP_KS,
        help="retrieved_k values to sweep in --k-sweep mode (default 100..2000).",
    )
    p.add_argument(
        "--rebuild", action="store_true",
        help="Offline: re-render table.md from an existing hybrid_bounds_summary.json "
             "(re-injects gold-flip units + per-metric verdict, fixes the alpha label). "
             "No Weaviate. R@5/R@20 stay blank until a live run repopulates them.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Where to write hybrid_bounds_{summary.json,table.md,errorbars.json}.",
    )
    return p.parse_args()


def rebuild_from_summary(output_dir: Path) -> None:
    """Re-derive the new-format table from an on-disk summary.json (no network).

    The persisted summary stores per-cell mean/min/max/range/std but not the
    derived presentation fields (q, gold_flips, per_metric_verdict), and may
    carry a stale alpha label — so we recompute those from the stored values and
    rewrite table.md (and refresh summary.json in place). Ranked metrics absent
    from the old summary (R@5/R@20) simply render as `n/a` — the honest signal
    that a live re-run is still needed to fill them.
    """
    summary_path = output_dir / "hybrid_bounds_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"--rebuild: no summary at {summary_path}; run a live pass first.")
    with open(summary_path) as f:
        summary = json.load(f)

    # Correct the known-stale alpha label (was 0.5; actual unset Weaviate default).
    cfg = summary.setdefault("config", {})
    cfg["alpha"] = 0.75
    cfg.setdefault("alpha_note",
                   "weaviate server default; not set client-side (hybrid_alpha=None)")

    # Re-inject gold-flip units into every cell that has a known Q.
    for family in ("ranked", "pool_recall"):
        for subset, cells in summary.get(family, {}).get("per_subset", {}).items():
            q = SUBSET_QUERY_COUNTS.get(subset)
            if not q:
                continue
            for c in cells.values():
                c["q"] = q
                c["gold_flips"] = c.get("range", 0.0) * q
    summary["ranked"]["per_metric_verdict"] = _per_metric_verdict(
        summary["ranked"]["per_subset"]
    )

    table = build_table(summary)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "hybrid_bounds_table.md", "w") as f:
        f.write(table)
    print(table)
    print(f"\nRebuilt {output_dir / 'hybrid_bounds_table.md'} (offline, from summary.json)")
    print("NOTE: R@5/R@20 show n/a — re-run live to populate them.")


def main() -> None:
    args = parse_args()
    import os

    if args.rebuild:
        rebuild_from_summary(args.output_dir)
        return

    # Sanitize the Weaviate creds: a stray trailing newline (common from
    # `export X=$(cat file)` or a copy-paste) lands inside the connection URL and
    # raises `InvalidURL: ... '\n' at position 8`. qab + BaseRetriever read these
    # straight from os.environ at connect time, so strip in-place so both see the
    # cleaned value.
    for var in ("WEAVIATE_URL", "WEAVIATE_API_KEY"):
        val = os.getenv(var)
        if not val:
            raise SystemExit(f"Missing required env var: {var}")
        cleaned = val.strip()
        if cleaned != val:
            os.environ[var] = cleaned
            print(f"note: stripped surrounding whitespace/newline from {var}")

    if args.k_sweep:
        config = {
            "mode": "k_sweep",
            "n_trials": args.n_trials,
            "sweep_ks": args.sweep_ks,
            "alpha": 0.75,
            "alpha_note": "weaviate server default; not set client-side (hybrid_alpha=None)",
            "fusion": "relative_score",
            "first_stage": "weaviate hybrid (BM25 + Arctic 2.0)",
            "seed_policy": "no client-side seeding; natural query-time variation",
            "random_seed": RANDOM_SEED,
            "subsets": list(args.datasets),
            "sweep_metrics": list(SWEEP_METRICS.keys()),
            "std_ddof": 1,
            "num_samples": args.num_samples,
        }
        sweep_trials = run_k_sweep(
            args.datasets,
            sweep_ks=args.sweep_ks,
            n_trials=args.n_trials,
            num_samples=args.num_samples,
            max_concurrent=args.max_concurrent,
            use_async=not args.sync,
        )
        sweep_summary = build_sweep_summary(sweep_trials, args.sweep_ks, config)
        sweep_table = build_sweep_table(sweep_summary)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with open(args.output_dir / "hybrid_bounds_vs_k.json", "w") as f:
            json.dump(sweep_summary, f, indent=2)
        with open(args.output_dir / "hybrid_bounds_vs_k.md", "w") as f:
            f.write(sweep_table)
        print("\n" + sweep_table)
        print(f"Wrote {args.output_dir / 'hybrid_bounds_vs_k.json'}")
        print(f"Wrote {args.output_dir / 'hybrid_bounds_vs_k.md'}")
        return

    per_subset_trials: dict[str, dict[str, list[float]]] = {}
    for subset in args.datasets:
        try:
            per_subset_trials[subset] = run_subset(
                subset,
                n_trials=args.n_trials,
                retrieved_k=args.retrieved_k,
                num_samples=args.num_samples,
                max_concurrent=args.max_concurrent,
                use_async=not args.sync,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED {subset}: {e!r}")

    if not per_subset_trials:
        raise SystemExit("No subsets produced results; nothing to summarize.")

    config = {
        "n_trials": args.n_trials,
        "retrieved_k": args.retrieved_k,
        # alpha is NOT set client-side (BaseRetriever hybrid_alpha=None), so the
        # Weaviate server default applies — 0.75, relativeScoreFusion. This is
        # the same setting the paper's hybrid_only uses; report it, don't assert 0.5.
        "alpha": 0.75,
        "alpha_note": "weaviate server default; not set client-side (hybrid_alpha=None)",
        "fusion": "relative_score",
        "embedding": "weaviate/Snowflake/snowflake-arctic-embed-l-v2.0",
        "first_stage": "weaviate hybrid (BM25 + Arctic 2.0)",
        "seed_policy": "no client-side seeding; natural query-time variation",
        "random_seed": RANDOM_SEED,
        "subsets": list(per_subset_trials.keys()),
        "pool_recall_ks": POOL_RECALL_KS,
        "ranked_metrics": list(RANKED_METRICS.keys()),
        "std_ddof": 1,
        "num_samples": args.num_samples,
    }

    summary = build_summary(per_subset_trials, config)
    table = build_table(summary)
    errorbars = build_errorbars(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "hybrid_bounds_summary.json"
    table_path = args.output_dir / "hybrid_bounds_table.md"
    errorbars_path = args.output_dir / "hybrid_bounds_errorbars.json"

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(table_path, "w") as f:
        f.write(table)
    with open(errorbars_path, "w") as f:
        json.dump(errorbars, f, indent=2)

    print("\n" + table)
    print(f"Wrote {summary_path}")
    print(f"Wrote {table_path}")
    print(f"Wrote {errorbars_path}")


if __name__ == "__main__":
    main()
