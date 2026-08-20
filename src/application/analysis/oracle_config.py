"""Oracle-config decomposition: routing value vs blending value, swept over k.

Quantifies how much of the per-query reranking optimum comes from ROUTING
(choosing the single best singleton per query) versus BLENDING (choosing an
interior fusion of several rerankers per query). Fully --k-parameterized; the
paper sweeps retrieved_k in {100, 200, 500, 1000, 2000}, each writing its own
oracle_config_k{N}.{json,_table.md}. retrieved_k=200 is the clean reference
operating point (Cohere healthy). This is the empirical core of the §5.1
routing-vs-blending argument. (The file name still says k200 for historical
reasons; it is a general k sweep — see the --k flag.)

Three per-query reference quantities, all over the same active condition menu
(equal-weight only) the main experiment runs (3 singletons + the equal-weight
RRF/RSF fusion blends; excludes hybrid_only):

  1. best static fusion — ONE fixed fusion blend, chosen once as the argmax of
     the per-query median of a primary metric over the blends, then applied
     to every query. Selected GLOBALLY (one blend across all five subsets) for
     the headline; a per-subset-tuned variant is reported in an appendix. Per
     the resolved spec decision, a SINGLE config anchors every metric row (the
     blend that wins the primary selection metric, SELECTION_METRIC), so the
     routing-value column is "oracle-selector minus the one blend you deploy."
  2. oracle-selector — per query, the best of the three singletons. Pure
     routing: a perfect per-query router constrained to one model's list.
  3. oracle-config  — per query, the best of every condition in the menu. Adds
     the freedom to choose an interior blend per query.

Two gaps:
  routing value  = median(oracle-selector) - median(best static fusion)
  blending value = median(oracle-config)   - median(oracle-selector)

DERIVE-ONCE CORRECTNESS. Every condition at k=200 is materialized by
DerivedSearchAgent from the cached k=2000 scores, restricted to the top-200
hybrid pool. That path recomputes RRF ranks within the 200-doc set and re-does
RSF min-max normalization ON THE RESTRICTED 200-doc pool (never inheriting the
k=2000 normalization) — the single most common way this analysis goes silently
wrong. Reusing DerivedSearchAgent (rather than re-implementing the sort)
guarantees the rankings are byte-identical to the runs/ derive path. Zero
reranker API calls: this module never imports a provider client.

Aggregation is MEAN across queries per subset, and MEDIAN across subsets for
the aggregate row (each quantity aggregated independently). NOTE: the spec said
"median across queries", but per-query recall@K is too coarse for a median —
for most queries it is exactly 0, otherwise 1/|gold| — so a query-median
collapses to a single representative query's value and can never reproduce the
smooth blending-value bands the spec's own acceptance criteria cite (those are
means, as is every other number in this experiment and agreement_analysis.py).
We therefore aggregate across queries with the MEAN; median is retained only
for the cross-subset aggregate, where 5 subset-level means make it sensible.
The query universe
per subset is the all-three-present intersection (the same set
agreement_analysis uses), so every condition in the menu is defined for
every query and the per-query max is clean — guaranteeing
oracle_config >= oracle_selector >= best singleton, per query and per median.

Outputs (under results/):
  oracle_config_k{K}.json        — machine-readable decomposition
  oracle_config_k{K}_table.md    — the figure-replacing table (one block/metric)

Usage:
    uv run python scripts/oracle_config.py
    uv run python scripts/oracle_config.py --k 2000   # 5.2 toggle
    uv run python scripts/oracle_config.py --smoke
"""
from __future__ import annotations

import argparse
import json
import statistics
from typing import Optional

from src.application.derived import DerivedSearchAgent  # noqa: F401
from src.config import MODEL_OVERRIDES, RESULTS_DIR
from src.domain.conditions import (  # noqa: F401
    CONDITIONS as _RE_CONDITIONS,
    SINGLETON_CONDITIONS,
)
from src.application.decompose import (
    FUSION_CONFIGS,
    METRIC_LABEL,
    ORACLE_CONFIG_MENU,
    SELECTION_METRIC,
    SINGLETONS,
    _decompose,
    _neg_name,  # noqa: F401
    _per_query_max,
    _qmean,
    per_query_condition_metrics,
    select_best_static_fusion,
)
from src.domain.metrics import CAP20_METRICS, metric as _metric  # noqa: F401
from src.application.queryset import load_and_validate

from src.adapters import qab

qab.setup()

# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #

RETRIEVED_K = 200            # clean operating point (Cohere healthy; see spec).
RERANKED_K = 20              # deployed output cap; top-20 covers R@1/5/20/nDCG@10.
CACHE_K = 2000               # provenance: derive everything from the k=2000 cache.

# Subsets in canonical order (spec Procedure).
SUBSETS = ["biology", "earth_science", "economics", "psychology", "robotics"]


# --------------------------------------------------------------------------- #
# Invariants (acceptance criteria)                                            #
# --------------------------------------------------------------------------- #


def check_invariants(
    table: dict, n: int, bsf_config: str, menu=ORACLE_CONFIG_MENU, singletons=SINGLETONS
) -> list[str]:
    """oracle_config >= oracle_selector >= each singleton; >= best static fusion.

    Checked on the query means (which inherit the per-query domination because
    the per-query max pointwise-dominates each component, and the mean is
    monotone under pointwise domination). `menu`/`singletons` default to the
    full menu; pass restricted lists to check a sub-menu (noise_null.py).
    """
    errs: list[str] = []
    singleton_names = [c.name for c in singletons]
    menu_names = [c.name for c in menu]
    for m in CAP20_METRICS:
        sel = _qmean(_per_query_max(table, singleton_names, m, n))
        cfg = _qmean(_per_query_max(table, menu_names, m, n))
        bsf = _qmean(table[bsf_config][m])
        if cfg < sel - 1e-9:
            errs.append(f"oracle_config {m} {cfg:.4f} < oracle_selector {sel:.4f}")
        if cfg < bsf - 1e-9:
            errs.append(f"oracle_config {m} {cfg:.4f} < best_static_fusion {bsf:.4f}")
        for s in singleton_names:
            smean = _qmean(table[s][m])
            if sel < smean - 1e-9:
                errs.append(f"oracle_selector {m} {sel:.4f} < singleton {s} {smean:.4f}")
    return errs


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #


def run(k: int, reranked_k: int, write: bool = True) -> dict:
    """Compute the decomposition for every subset at retrieved_k=k."""
    # 1. Materialize per-query per-condition metrics for each subset.
    tables: dict[str, dict] = {}
    queries_by_subset: dict[str, list[str]] = {}
    drops_by_subset: dict[str, dict] = {}
    for ds in SUBSETS:
        print(f"\n=== {ds} (k={k}) ===")
        loaded = load_and_validate(ds)
        if loaded is None:
            print(f"  [skip] no usable cache for {ds}")
            continue
        cache, qs = loaded
        table, queries = per_query_condition_metrics(cache, qs, k, reranked_k)
        tables[ds] = table
        queries_by_subset[ds] = queries
        drops_by_subset[ds] = qs.drops
        print(f"  {len(queries)} intersection queries scored over {len(ORACLE_CONFIG_MENU)} conditions")

    present = [ds for ds in SUBSETS if ds in tables]
    if not present:
        raise SystemExit("No subsets had a usable k=2000 cache.")

    # 2. GLOBAL best static fusion: argmax over the fusion blends of the median of
    #    SELECTION_METRIC over the POOLED query set (all subsets concatenated).
    pooled_sel: dict[str, list[float]] = {
        c.name: [] for c in FUSION_CONFIGS
    }
    for ds in present:
        for c in FUSION_CONFIGS:
            pooled_sel[c.name].extend(tables[ds][c.name][SELECTION_METRIC])
    global_config, global_median = select_best_static_fusion(pooled_sel)
    print(
        f"\nGlobal best static fusion (by mean {SELECTION_METRIC} over pooled "
        f"queries): {global_config} (mean={global_median:.4f})"
    )

    # 3. Per-subset decomposition with the GLOBAL config (headline) + per-subset
    #    tuned config (appendix). Also run invariant checks per subset.
    per_subset: dict[str, dict] = {}
    per_subset_tuned: dict[str, dict] = {}
    all_invariant_errors: dict[str, list[str]] = {}
    for ds in present:
        n = len(queries_by_subset[ds])
        per_subset[ds] = _decompose(tables[ds], n, global_config)
        # Per-subset-tuned config: argmax within this subset.
        sub_sel = {c.name: tables[ds][c.name][SELECTION_METRIC] for c in FUSION_CONFIGS}
        tuned_config, tuned_median = select_best_static_fusion(sub_sel)
        per_subset_tuned[ds] = {
            "config": tuned_config,
            f"mean_{SELECTION_METRIC}": tuned_median,
            "decomposition": _decompose(tables[ds], n, tuned_config),
        }
        errs = check_invariants(tables[ds], n, global_config)
        if errs:
            all_invariant_errors[ds] = errs
            print(f"  [WARN] invariant errors ({ds}): {errs}")

    # 4. Aggregate: median across subsets of each quantity, per metric (global
    #    config headline).
    quantities = (
        "best_static_fusion",
        "oracle_selector",
        "oracle_config",
        "routing_value",
        "blending_value",
    )
    aggregate: dict[str, dict[str, float]] = {}
    for m in CAP20_METRICS:
        label = METRIC_LABEL[m]
        aggregate[label] = {
            q: statistics.median([per_subset[ds][label][q] for ds in present])
            for q in quantities
        }

    payload = {
        "k": k,
        "reranked_k": reranked_k,
        "cache_retrieved_k": CACHE_K,
        "model_overrides": MODEL_OVERRIDES,
        "best_static_fusion": {
            "selection": "global",
            "selection_metric": SELECTION_METRIC,
            "config": global_config,
            f"mean_{SELECTION_METRIC}_pooled": global_median,
            "note": (
                "ONE fusion blend, chosen once as the argmax over the fusion blends "
                f"of the pooled-query mean of {SELECTION_METRIC}, applied to "
                "every query and every metric row. Per-subset-tuned configs are "
                "in best_static_fusion_per_subset (appendix / domain-tuning "
                "ceiling)."
            ),
        },
        "menu": {
            "n_conditions": len(ORACLE_CONFIG_MENU),
            "singletons": [c.name for c in SINGLETONS],
            "n_fusion_blends": len(FUSION_CONFIGS),
        },
        "subsets": {
            ds: {
                "n_queries_intersection": len(queries_by_subset[ds]),
                "drops": drops_by_subset[ds],
            }
            for ds in present
        },
        "per_subset": per_subset,
        "aggregate": aggregate,
        "best_static_fusion_per_subset": per_subset_tuned,
        "invariant_errors": all_invariant_errors,
        "invariants_ok": not all_invariant_errors,
    }

    if write:
        out_dir = RESULTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"oracle_config_k{k}.json"
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote {json_path}")
        md_path = out_dir / f"oracle_config_k{k}_table.md"
        md_path.write_text(render_table(payload, present))
        print(f"Wrote {md_path}")

    return payload


# --------------------------------------------------------------------------- #
# Markdown table (figure-replacing)                                           #
# --------------------------------------------------------------------------- #


def _f(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def render_table(payload: dict, present: list[str]) -> str:
    k = payload["k"]
    bsf = payload["best_static_fusion"]
    lines: list[str] = []
    A = lines.append

    A(f"# Oracle-Config Decomposition at k={k}")
    A("")
    A(
        f"Routing vs blending decomposition over the {payload['menu']['n_conditions']}-"
        f"condition menu ({len(payload['menu']['singletons'])} singletons + "
        f"{payload['menu']['n_fusion_blends']} fusion blends), at retrieved_k={k}, "
        f"reranked_k={payload['reranked_k']}, derived from the k="
        f"{payload['cache_retrieved_k']} score cache with zero reranker API calls "
        "(RSF normalization recomputed on the restricted pool)."
    )
    A("")
    A(
        f"**best static fusion**: `{bsf['config']}` — one fixed blend, selected "
        f"**{bsf['selection']}ly** as the argmax over the fusion blends of the pooled-"
        f"query mean of `{bsf['selection_metric']}`, applied to every query and "
        "every metric row (single-config deployment baseline)."
    )
    A("")
    A(
        "- **oracle-selector** = per-query best of the 3 singletons (pure routing). "
        "- **oracle-config** = per-query best of every condition in the menu "
        "(adds interior blending)."
    )
    A(
        "- **routing value** = median(oracle-selector) − median(best static fusion). "
        "- **blending value** = median(oracle-config) − median(oracle-selector)."
    )
    A("")
    A(
        "Per-subset values are means across that subset's intersection queries; "
        "the **aggregate** row is the median across subsets of each quantity "
        "(quantities aggregated independently). The query-level statistic is the "
        "mean, not a median: per-query recall@K is too coarse for a median (mostly "
        "0, else 1/|gold|), which collapses to a single query's value."
    )
    A("")
    A("Query counts (all-three-present intersection): "
      + ", ".join(f"{ds} {payload['subsets'][ds]['n_queries_intersection']}" for ds in present)
      + ".")
    A("")

    cols = ["best static fusion", "oracle-selector", "oracle-config",
            "routing value", "blending value"]
    keys = ["best_static_fusion", "oracle_selector", "oracle_config",
            "routing_value", "blending_value"]

    for m in CAP20_METRICS:
        label = METRIC_LABEL[m]
        A(f"## {label}")
        A("")
        A("| Subset | " + " | ".join(cols) + " |")
        A("|---|" + "|".join(["---"] * len(cols)) + "|")
        for ds in present:
            row = payload["per_subset"][ds][label]
            A(f"| {ds} | " + " | ".join(_f(row[kk]) for kk in keys) + " |")
        agg = payload["aggregate"][label]
        A("| **aggregate** | " + " | ".join(f"**{_f(agg[kk])}**" for kk in keys) + " |")
        A("")

    # Appendix: per-subset-tuned best static fusion (domain-tuning ceiling).
    A("## Appendix: per-subset-tuned best static fusion")
    A("")
    A(
        "Each subset picks its OWN best fixed blend (argmax of in-subset mean "
        f"`{bsf['selection_metric']}`). This is the ceiling of domain-specific "
        "static tuning — not deployable at test time without per-domain selection, "
        "but it bounds how much the global single-config choice leaves on the table."
    )
    A("")
    A("| Subset | tuned config | "
      + " | ".join(f"routing/blending {METRIC_LABEL[m]}" for m in CAP20_METRICS) + " |")
    A("|---|---|" + "|".join(["---"] * len(CAP20_METRICS)) + "|")
    for ds in present:
        t = payload["best_static_fusion_per_subset"][ds]
        cells = []
        for m in CAP20_METRICS:
            d = t["decomposition"][METRIC_LABEL[m]]
            cells.append(f"{_f(d['routing_value'])} / {_f(d['blending_value'])}")
        A(f"| {ds} | `{t['config']}` | " + " | ".join(cells) + " |")
    A("")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def run_smoke() -> None:
    """biology only, first 10 intersection queries, k=200; assert invariants."""
    print("=== SMOKE: biology k=200, first 10 queries ===")
    loaded = load_and_validate("biology")
    if loaded is None:
        raise SystemExit("Smoke needs the biology k=2000 cache.")
    cache, qs = loaded
    keep = list(qs.gold.keys())[:10]
    qs.gold = {t: qs.gold[t] for t in keep}
    table, queries = per_query_condition_metrics(cache, qs, RETRIEVED_K, RERANKED_K)
    n = len(queries)
    sub_sel = {c.name: table[c.name][SELECTION_METRIC] for c in FUSION_CONFIGS}
    config, med = select_best_static_fusion(sub_sel)
    errs = check_invariants(table, n, config)
    assert not errs, errs
    decomp = _decompose(table, n, config)
    print(f"  n={n}  best_static_fusion={config} (mean {SELECTION_METRIC}={med:.4f})")
    for m in CAP20_METRICS:
        d = decomp[METRIC_LABEL[m]]
        print(
            f"  {METRIC_LABEL[m]:>8}: bsf={d['best_static_fusion']:.3f} "
            f"sel={d['oracle_selector']:.3f} cfg={d['oracle_config']:.3f} "
            f"routing={d['routing_value']:+.3f} blending={d['blending_value']:+.3f}"
        )
    print("\nSMOKE PASSED.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--k", type=int, default=RETRIEVED_K,
        help="retrieved_k operating point (default 200; pass 2000 for the 5.2 toggle).",
    )
    parser.add_argument(
        "--reranked-k", type=int, default=RERANKED_K,
        help="Output cap for the rankings (default 20; covers R@1/5/20/nDCG@10).",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        run_smoke()
        return
    payload = run(args.k, args.reranked_k)
    if not payload["invariants_ok"]:
        print(f"\n[WARN] invariant errors: {payload['invariant_errors']}")


if __name__ == "__main__":
    main()
