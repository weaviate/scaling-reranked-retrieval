"""Per-singleton deep-recall grid: R@{1,5,20,50,100} × full retrieved_k sweep.

Closes the gap noted in CLAUDE.md: R@1 and R@20 for each standalone reranker
(`cohere`, `voyage`, `zerank`) are tabulated across the k sweep, but R@50 and
R@100 per singleton appear only as sparse winner cells — never as a full grid.
This produces the complete deep-recall scaling curve for each singleton so it
can be plotted alongside the existing R@1/R@20 lines on one shared axis.

READ-ONLY, ZERO-API. Every cell is derived from the k=2000 score caches via
`DerivedSearchAgent` — the same path agreement_analysis / oracle_config_k200 /
unique_successes_k200 use. A startup guard wraps the reranker client factories
in the retrieval adapters so that constructing ANY reranker client raises (acceptance
criterion: no reranker client is ever constructed). Singletons sort cached
scores directly — the singleton path in DerivedSearchAgent bypasses fusion
entirely, so there is no RSF pool-restricted normalization and no
`PYTHONHASHSEED` tie-break exposure here (distinct-float sort is deterministic).

DENOMINATOR (the key methodology choice). Recall is computed over the
ALL-THREE-PRESENT intersection (queries scored by cohere AND voyage AND zerank
at k=2000), identical to agreement_analysis.build_query_set. This is what keeps
the new R@5/R@50/R@100 lines on the SAME footing as the R@1/R@20 lines already
plotted in §5.1 — one denominator, one axis, no seam.

  WHY NOT read R@50/R@100 straight from runs_rk100/: those files average each
  singleton over its OWN coverage denominator (the queries that provider could
  score), not the intersection. Reading them would put R@50/R@100 on a
  different denominator than the intersection-based R@1/R@20 — exactly the seam
  this extraction exists to avoid. So every singleton cell is RE-DERIVED over
  the intersection from caches/k2000.json (provenance: "derived_from_cache"),
  and the runs_rk100/ own-coverage values are reported only as a soft
  cross-check (typically within ~1 query of the intersection numbers).

REGRESSION GUARD (the spec lists two value sets; they live on two different
denominators — see CLAUDE.md "Best-singleton sourcing" / equal_weight
"Provenance / don't conflate"):
  - HARD (±0.002): the all-5 intersection best-singleton line reproduces §5.1
    EXACTLY — R@1 0.353/0.414/0.354/0.374/0.364, R@20 0.395/0.468/0.565/0.590/
    0.599. These are produced by the same intersection + DerivedSearchAgent
    path this script uses, so they must match to float noise. Failure here
    means the derive path is wrong → nonzero exit.
  - SOFT (~1 query): the published all-5 per-reranker and best-singleton
    lines (cohere 0.301/0.301/0.330/0.263/0.212, best-singleton 0.340/0.414/
    0.366/0.386/0.376, etc.) come from equal_weight_compare reading runs*/ over
    each condition's OWN coverage. The intersection cannot reproduce them to
    ±0.002 (they differ by the ~1-query intersection-vs-coverage gap CLAUDE.md
    documents); we print the deltas and warn only if a delta exceeds ~1 query.

Aggregation: MEAN across queries per subset (per-query recall@K is too coarse
for a median — mostly 0, else 1/|gold|), MEDIAN across all five subsets for the
cross-subset line. best_singleton per (k, cutoff)
is the per-subset MAX over the three rerankers, THEN median across subsets
(max-then-median — the §5.1 definition; this is what reproduces the guard
line, not max-of-medians).

Outputs (under results/):
  singleton_deep_recall.json        — full grid + cross-subset medians +
                                       best_singleton + provenance + guards
  singleton_deep_recall_table.md    — cutoff×k matrices per reranker (all-5
                                       cross-subset median) + best_singleton

Usage:
    uv run python scripts/singleton_deep_recall.py
    uv run python scripts/singleton_deep_recall.py --k 2000
    uv run python scripts/singleton_deep_recall.py --subsets biology economics
    uv run python scripts/singleton_deep_recall.py --smoke
"""
from __future__ import annotations

import argparse
import json
import statistics
from typing import Optional

# --------------------------------------------------------------------------- #
# Zero-API guard: poison the reranker client factories so any attempt to       #
# construct a reranker client fails loudly. This is read-only by construction  #
# (we only call DerivedSearchAgent), but the guard makes the acceptance        #
# criterion enforceable rather than aspirational.                              #
# --------------------------------------------------------------------------- #


def _install_no_reranker_client_guard() -> None:
    import src.adapters.retrieval.clients as _clients

    def _poison(name):
        def _raise(*_a, **_k):
            raise AssertionError(
                f"singleton_deep_recall is read-only (zero-API): refusing to "
                f"construct a reranker client via {name}(). All cells derive "
                "from caches/k2000.json through DerivedSearchAgent."
            )

        return _raise

    for _name in (
        "get_cohere_client",
        "get_cohere_async_client",
        "get_voyage_client",
        "get_voyage_async_client",
        "get_zerank_client",
        "get_zerank_async_client",
    ):
        if hasattr(_clients, _name):
            setattr(_clients, _name, _poison(_name))


_install_no_reranker_client_guard()

from src.application.derived import DerivedSearchAgent  # noqa: E402
from src.domain.conditions import _Condition  # noqa: E402
from src.config import (  # noqa: E402
    CACHE_K,
    MODEL_OVERRIDES,
    PROVIDERS,
    RESULTS_DIR,
    get_results_dir,
)
from src.domain.metrics import metric as _metric  # noqa: E402
from src.application.queryset import load_and_validate  # noqa: E402

from src.adapters import qab  # noqa: E402

qab.setup()

# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #

# Reranker name == provider name for singletons (cohere/voyage/zerank).
RERANKERS = list(PROVIDERS)

K_VALUES = (100, 200, 500, 1000, 2000)
# Output cap 100 so R@50/R@100 are measurable. R@1/R@5/R@20 are byte-identical
# to a cap-20 run (top-20 of a reranked top-100 == top-20 reranked directly,
# cross-encoder scoring being per-(query,doc) independent), so this single cap
# yields all five cutoffs on one path.
RERANKED_K = 100

# The five cutoffs, in order, as runs-file metric keys.
CUTOFFS = ("recall_at_1", "recall_at_5", "recall_at_20", "recall_at_50", "recall_at_100")

SUBSETS = ["biology", "earth_science", "economics", "psychology", "robotics"]

# Published all-three-present intersection sizes (CLAUDE.md / §5.1). Confirmed
# at runtime; a mismatch is surfaced (not necessarily fatal — a subset may be
# absent or a cache may have changed).
PUBLISHED_INTERSECTION_N = {
    "biology": 102, "earth_science": 114, "economics": 103,
    "psychology": 99, "robotics": 92,
}

# --- Regression-guard target lines (across k = 100/200/500/1000/2000) ------- #

# HARD guard (±0.002): all-5 intersection best-singleton — the §5.1 line. This
# is produced by THIS script's exact path (intersection + DerivedSearchAgent +
# per-subset-max-then-median), so it must reproduce to float noise.
GUARD_HARD_ALL5_BEST_SINGLETON = {
    "recall_at_1": [0.353, 0.414, 0.354, 0.374, 0.364],
    "recall_at_20": [0.395, 0.468, 0.565, 0.590, 0.599],
}
GUARD_HARD_TOL = 0.002

# SOFT cross-check (~1 query): published ALL-5 per-reranker + best-singleton
# lines. These are equal_weight_compare's OWN-COVERAGE numbers, on a
# different denominator than the intersection — so we expect agreement only to
# ~1 query and warn solely on larger drift.
GUARD_SOFT_ALL5_PER_RERANKER = {
    "recall_at_1": {
        "cohere": [0.301, 0.301, 0.330, 0.263, 0.212],
        "voyage": [0.340, 0.376, 0.356, 0.356, 0.356],
        "zerank": [0.311, 0.386, 0.366, 0.386, 0.376],
    },
    "recall_at_20": {
        "cohere": [0.368, 0.467, 0.544, 0.560, 0.536],
        "voyage": [0.367, 0.465, 0.531, 0.549, 0.549],
        "zerank": [0.395, 0.468, 0.560, 0.588, 0.597],
    },
}
GUARD_SOFT_ALL5_BEST_SINGLETON = {
    "recall_at_1": [0.340, 0.414, 0.366, 0.386, 0.376],
    "recall_at_20": [0.395, 0.468, 0.564, 0.588, 0.597],
}
# ~1 query at the smallest intersection (~92): allow ~1.5 queries of slack
# before warning.
GUARD_SOFT_TOL = 0.018

METRIC_LABEL = {
    "recall_at_1": "R@1", "recall_at_5": "R@5", "recall_at_20": "R@20",
    "recall_at_50": "R@50", "recall_at_100": "R@100",
}


# --------------------------------------------------------------------------- #
# Per-subset derivation (intersection, DerivedSearchAgent)                      #
# --------------------------------------------------------------------------- #


def compute_subset(cache, qs, ks: list[int]) -> tuple[dict, int]:
    """Per (reranker, k, cutoff) MEAN recall over the intersection queries.

    Returns {reranker: {k: {cutoff: mean}}}. One DerivedSearchAgent run per
    (query, reranker, k) at reranked_k=RERANKED_K; all five cutoffs read off the
    same reranked list (qab's recall@k slices internally).
    """
    queries = list(qs.gold.keys())
    n = len(queries)
    out = {r: {k: {c: 0.0 for c in CUTOFFS} for k in ks} for r in RERANKERS}
    for k in ks:
        for r in RERANKERS:
            sums = {c: 0.0 for c in CUTOFFS}
            for text in queries:
                gold_list = list(qs.gold[text])
                agent = DerivedSearchAgent(
                    cache=cache, retrieved_k=k,
                    condition=_Condition(provider=r), reranked_k=RERANKED_K,
                )
                ranked = [o.object_id for o in agent.run(text)]
                for c in CUTOFFS:
                    sums[c] += _metric(c, gold_list, ranked)
            for c in CUTOFFS:
                out[r][k][c] = sums[c] / n if n else 0.0
    return out, n


def _mechanical(k: int, cutoff: str) -> bool:
    """k=100 R@100 is the mechanical identity.

    At retrieved_k=k with reranked_k=100 >= k, the singleton emits the entire
    k-doc pool (reordered), so recall@K for K>=k equals the hybrid R@K ceiling —
    not a reranker achievement, just the pool's own recall. In this grid that is
    exactly (k=100, recall@100). (recall@50 at k=100 still depends on the
    reranked top-50, so it is NOT mechanical.)
    """
    cut_k = int(cutoff.partition("_at_")[2])
    return cut_k >= k and RERANKED_K >= k


# --------------------------------------------------------------------------- #
# runs_rk100 own-coverage cross-check (soft; NOT the source of any grid cell)   #
# --------------------------------------------------------------------------- #


def read_runs_rk100_singletons(dataset_slug: str, k: int) -> dict:
    """Own-coverage R@{50,100} (and R@1/20) for the singletons from runs_rk100/.

    Returns {reranker: {cutoff: value or None}}. Used ONLY to cross-check the
    intersection-derived cells, never as their source (different denominator).
    """
    rd = get_results_dir(dataset_slug)
    path = rd / "runs_rk100" / f"k{k}_from_k{CACHE_K}.json"
    out = {r: {c: None for c in CUTOFFS} for r in RERANKERS}
    if not path.exists():
        return out
    with open(path) as f:
        results = json.load(f).get("results", {})
    for r in RERANKERS:
        entry = results.get(f"{r}_only")
        if not entry or "error" in entry:
            continue
        for c in CUTOFFS:
            out[r][c] = entry.get(f"avg_{c}_mean")
    return out


# --------------------------------------------------------------------------- #
# Cross-subset aggregation                                                      #
# --------------------------------------------------------------------------- #


def _median(xs: list[float]) -> float:
    return statistics.median(xs) if xs else 0.0


def cross_subset(per_subset: dict, subsets: list[str], ks: list[int]) -> dict:
    """Per (reranker, k, cutoff) median across the given subsets."""
    return {
        r: {
            str(k): {
                c: _median([per_subset[ds][r][k][c] for ds in subsets])
                for c in CUTOFFS
            }
            for k in ks
        }
        for r in RERANKERS
    }


def best_singleton(per_subset: dict, subsets: list[str], ks: list[int]) -> dict:
    """Per (k, cutoff): per-subset MAX over rerankers, THEN median across subsets.

    This max-then-median order is the §5.1 best_singleton definition and is what
    reproduces the regression-guard line (max-of-medians would differ).
    """
    return {
        str(k): {
            c: _median([
                max(per_subset[ds][r][k][c] for r in RERANKERS) for ds in subsets
            ])
            for c in CUTOFFS
        }
        for k in ks
    }


# --------------------------------------------------------------------------- #
# Regression guards                                                             #
# --------------------------------------------------------------------------- #


def run_guards(
    best_all5: dict, cross_all5: dict, ks: list[int],
) -> tuple[list[str], list[str]]:
    """Return (hard_errors, soft_warnings). Only run when the full k sweep is
    present (the published lines are full-sweep; a partial sweep can't match)."""
    hard: list[str] = []
    soft: list[str] = []
    if list(ks) != list(K_VALUES):
        soft.append("guards skipped: not the full k sweep")
        return hard, soft

    def at(line_for_k: dict, c: str) -> list[float]:
        return [line_for_k[str(k)][c] for k in K_VALUES]

    # HARD: all-5 best-singleton == §5.1 line (exact path → float noise).
    for c, target in GUARD_HARD_ALL5_BEST_SINGLETON.items():
        got = at(best_all5, c)
        for k, g, t in zip(K_VALUES, got, target):
            if abs(g - t) > GUARD_HARD_TOL:
                hard.append(
                    f"[HARD] all-5 best_singleton {METRIC_LABEL[c]} k={k}: "
                    f"got {g:.4f} vs §5.1 {t:.4f} (|Δ|={abs(g-t):.4f} > {GUARD_HARD_TOL})"
                )

    # SOFT: all-5 per-reranker vs published own-coverage equal_weight line.
    for c, by_r in GUARD_SOFT_ALL5_PER_RERANKER.items():
        for r, target in by_r.items():
            got = at(cross_all5[r], c)
            for k, g, t in zip(K_VALUES, got, target):
                if abs(g - t) > GUARD_SOFT_TOL:
                    soft.append(
                        f"[SOFT] all-5 {r} {METRIC_LABEL[c]} k={k}: intersection "
                        f"{g:.4f} vs published own-coverage {t:.4f} (Δ={g-t:+.4f})"
                    )
    # SOFT: all-5 best-singleton vs published own-coverage line.
    for c, target in GUARD_SOFT_ALL5_BEST_SINGLETON.items():
        got = at(best_all5, c)
        for k, g, t in zip(K_VALUES, got, target):
            if abs(g - t) > GUARD_SOFT_TOL:
                soft.append(
                    f"[SOFT] all-5 best_singleton {METRIC_LABEL[c]} k={k}: "
                    f"intersection {g:.4f} vs published own-coverage {t:.4f} (Δ={g-t:+.4f})"
                )
    return hard, soft


# --------------------------------------------------------------------------- #
# Driver                                                                        #
# --------------------------------------------------------------------------- #


def run(
    subsets: list[str], ks: list[int],
    write: bool = True,
) -> dict:
    per_subset: dict[str, dict] = {}
    intersection_counts: dict[str, int] = {}
    provenance: dict[str, dict] = {}
    crosscheck: dict[str, dict] = {}

    for ds in subsets:
        print(f"\n=== {ds} ===")
        loaded = load_and_validate(ds)
        if loaded is None:
            print(f"  [skip] no usable cache for {ds}")
            continue
        cache, qs = loaded
        sub, n = compute_subset(cache, qs, ks)
        per_subset[ds] = sub
        intersection_counts[ds] = n
        # Confirm against the published intersection size.
        exp = PUBLISHED_INTERSECTION_N.get(ds)
        flag = "" if exp is None or exp == n else f"  [WARN expected {exp}]"
        print(f"  intersection n={n}{flag}")
        # Provenance: every singleton cell is derived from the cache over the
        # intersection (runs_rk100 is own-coverage; see module docstring).
        provenance[ds] = {
            r: {str(k): {c: "derived_from_cache" for c in CUTOFFS} for k in ks}
            for r in RERANKERS
        }
        # Soft own-coverage cross-check from runs_rk100 (recorded, not sourced).
        crosscheck[ds] = {str(k): read_runs_rk100_singletons(ds, k) for k in ks}

    present = [ds for ds in subsets if ds in per_subset]
    if not present:
        raise SystemExit("No subsets had a usable k=2000 cache.")

    # Per-subset block with mechanical flags.
    per_subset_out = {}
    for ds in present:
        per_subset_out[ds] = {"rerankers": {}}
        for r in RERANKERS:
            per_subset_out[ds]["rerankers"][r] = {
                str(k): {
                    c: {
                        "value": per_subset[ds][r][k][c],
                        **({"mechanical": True} if _mechanical(k, c) else {}),
                    }
                    for c in CUTOFFS
                }
                for k in ks
            }

    cross_all5 = cross_subset(per_subset, present, ks)
    best_all5 = best_singleton(per_subset, present, ks)

    hard, soft = run_guards(best_all5, cross_all5, ks)
    for e in hard:
        print(f"  {e}")
    for w in soft:
        print(f"  {w}")

    payload = {
        "intersection_counts": intersection_counts,
        "cache_retrieved_k": CACHE_K,
        "reranked_k": RERANKED_K,
        "retrieved_k_sweep": list(ks),
        "cutoffs": list(CUTOFFS),
        "model_overrides": MODEL_OVERRIDES,
        "denominator": "all_three_present_intersection",
        "aggregation": "mean across queries (per subset); median across subsets",
        "notes": (
            "Every singleton cell is derived from caches/k2000.json via "
            "DerivedSearchAgent over the all-three-present intersection (the §5.1 "
            "denominator), NOT read from runs_rk100/ (which uses each condition's "
            "own coverage). This keeps R@1/R@5/R@20/R@50/R@100 on one denominator "
            "and one axis. runs_rk100 own-coverage values are recorded under "
            "runs_rk100_crosscheck for comparison only. k=100 R@100 cells are the "
            "mechanical hybrid-R@100 identity (flagged mechanical=True)."
        ),
        "per_subset": per_subset_out,
        "cross_subset_median": {
            "all5": cross_all5,
        },
        "best_singleton": {
            "all5": best_all5,
            "definition": "per-subset max over the 3 rerankers, then median across subsets",
        },
        "source_provenance": provenance,
        "runs_rk100_crosscheck": crosscheck,
        "regression_guard": {
            "hard_errors": hard,
            "soft_warnings": soft,
            "hard_ok": not hard,
            "hard_target_all5_best_singleton": GUARD_HARD_ALL5_BEST_SINGLETON,
            "hard_tol": GUARD_HARD_TOL,
            "soft_tol": GUARD_SOFT_TOL,
        },
    }

    if write:
        out_dir = RESULTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "singleton_deep_recall.json"
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote {json_path}")
        md_path = out_dir / "singleton_deep_recall_table.md"
        md_path.write_text(render_table(payload, present, ks))
        print(f"Wrote {md_path}")

    return payload


# --------------------------------------------------------------------------- #
# Markdown                                                                      #
# --------------------------------------------------------------------------- #


def _f(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _grid_block(A, title, line_for_k, ks, mech_at=None):
    """Render one reranker's cutoff×k matrix. `mech_at(k,cutoff)->bool` flags
    mechanical cells with a trailing *."""
    A(f"#### {title}")
    A("")
    A("| cutoff \\ k | " + " | ".join(f"k={k}" for k in ks) + " |")
    A("|---|" + "|".join(["---"] * len(ks)) + "|")
    for c in CUTOFFS:
        cells = []
        for k in ks:
            v = line_for_k[str(k)][c]
            mark = " \\*" if mech_at and mech_at(k, c) else ""
            cells.append(f"{_f(v)}{mark}")
        A(f"| {METRIC_LABEL[c]} | " + " | ".join(cells) + " |")
    A("")


def render_table(payload, present, ks) -> str:
    lines: list[str] = []
    A = lines.append

    A("# Per-Singleton Deep-Recall Grid (R@1/5/20/50/100 × retrieved_k)")
    A("")
    A(
        "Recall at every cutoff K ∈ {1, 5, 20, 50, 100} for each standalone "
        f"reranker across retrieved_k ∈ {{{', '.join(str(k) for k in ks)}}}, "
        f"reranked_k={payload['reranked_k']}, derived from the k="
        f"{payload['cache_retrieved_k']} score caches with **zero reranker-API "
        "calls** (DerivedSearchAgent; a startup guard makes constructing any "
        "reranker client raise)."
    )
    A("")
    A(
        "**One denominator, one axis.** Every cell is computed over the "
        "all-three-present intersection (queries scored by cohere AND voyage AND "
        "zerank), the SAME intersection + DerivedSearchAgent path as §5.1 — so "
        "all four cutoff lines (R@1/R@20 already plotted, plus the new "
        "R@5/R@50/R@100) share one denominator and can be plotted on one axis. "
        "Cells are NOT read from runs_rk100/ (which averages over each "
        "condition's own coverage); the own-coverage values are in the JSON's "
        "`runs_rk100_crosscheck` for comparison only."
    )
    A("")
    A(
        "Aggregation: MEAN across queries per subset, MEDIAN across subsets. "
        "Per-query recall@K is too coarse for a median (mostly 0, else 1/|gold|), "
        "so the query-level statistic is the mean. `best_singleton` = per-subset "
        "max over the three rerankers, then median across subsets (the §5.1 "
        "max-then-median order)."
    )
    A("")
    A("Intersection sizes: "
      + ", ".join(f"{ds} {payload['intersection_counts'][ds]}" for ds in present)
      + ".")
    A("")
    A("\\* = mechanical cell: at k=100 the reranked top-100 IS the 100-doc pool, "
      "so R@100 = the hybrid R@100 ceiling (a pool property, not a reranker "
      "achievement).")
    A("")

    mech = _mechanical

    # Primary: all-5 cross-subset median, per reranker.
    A("## Cross-subset median — ALL 5 (primary)")
    A("")
    A(f"Median across the {len(present)} subsets "
      f"({', '.join(present)}). Read the plot lines off this block.")
    A("")
    for r in RERANKERS:
        _grid_block(A, r, payload["cross_subset_median"]["all5"][r], ks, mech)
    _grid_block(A, "best singleton", payload["best_singleton"]["all5"], ks, mech)

    # Per-subset grids.
    A("## Per-subset grids")
    A("")
    for ds in present:
        A(f"### {ds} (n={payload['intersection_counts'][ds]})")
        A("")
        for r in RERANKERS:
            block = {
                str(k): {
                    c: payload["per_subset"][ds]["rerankers"][r][str(k)][c]["value"]
                    for c in CUTOFFS
                }
                for k in ks
            }
            _grid_block(A, r, block, ks, mech)

    # Regression guard summary.
    rg = payload["regression_guard"]
    A("## Regression guard")
    A("")
    A("- **HARD (±%.3f)** — all-5 best-singleton reproduces the §5.1 intersection "
      "line (R@1 0.353/0.414/0.354/0.374/0.364, R@20 0.395/0.468/0.565/0.590/0.599). "
      "Same path → must match to float noise. Status: **%s**."
      % (rg["hard_tol"], "PASS" if rg["hard_ok"] else "FAIL"))
    for e in rg["hard_errors"]:
        A(f"  - {e}")
    A("- **SOFT (~1 query, Δ printed)** — all-5 per-reranker + best-"
      "singleton vs the published own-coverage equal_weight lines. These live on "
      "a different denominator (own coverage vs intersection), so a ~1-query gap "
      "is expected, not a bug.")
    if rg["soft_warnings"]:
        for w in rg["soft_warnings"]:
            A(f"  - {w}")
    else:
        A("  - all soft cross-checks within ~1 query.")
    A("")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def run_smoke() -> None:
    """biology only, full k sweep, print to stdout, no write. Asserts basics."""
    print("=== SMOKE: biology, full k sweep ===")
    payload = run(["biology"], list(K_VALUES), write=False)
    n = payload["intersection_counts"]["biology"]
    assert n == PUBLISHED_INTERSECTION_N["biology"], f"biology n={n} != 102"
    bio = payload["per_subset"]["biology"]["rerankers"]
    # Monotonicity holds only among the proper-recall cutoffs (R@5 <= R@20 <=
    # R@50 <= R@100). qab's recall@1 is Success@1 (found_count, NOT divided by
    # |gold|), so R@1 can exceed R@5 — that is by design, not a bug.
    recall_cutoffs = ["recall_at_5", "recall_at_20", "recall_at_50", "recall_at_100"]
    for r in RERANKERS:
        for k in K_VALUES:
            vals = [bio[r][str(k)][c]["value"] for c in recall_cutoffs]
            assert all(a <= b + 1e-9 for a, b in zip(vals, vals[1:])), (
                f"non-monotone recall cutoffs for {r} k={k}: {vals}"
            )
    # Mechanical flag present exactly at k=100 R@100.
    assert bio["cohere"]["100"]["recall_at_100"].get("mechanical") is True
    assert "mechanical" not in bio["cohere"]["2000"]["recall_at_100"]
    print("\nSMOKE PASSED. biology single-subset medians:")
    for r in RERANKERS:
        line = [payload["cross_subset_median"]["all5"][r][str(k)]["recall_at_100"]
                for k in K_VALUES]
        print(f"  {r} R@100: " + "/".join(f"{x:.3f}" for x in line))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=None,
                        help="Single retrieved_k (default: full sweep 100/200/500/1000/2000).")
    parser.add_argument("--subsets", nargs="+", default=None, choices=SUBSETS,
                        help="Subset(s) to include (default: all five).")
    parser.add_argument("--smoke", action="store_true",
                        help="biology only, full k sweep, print to stdout, no write.")
    args = parser.parse_args()

    if args.smoke:
        run_smoke()
        return

    ks: list[int] = [args.k] if args.k else list(K_VALUES)
    subsets: list[str] = args.subsets if args.subsets else SUBSETS
    payload = run(subsets, ks)
    if not payload["regression_guard"]["hard_ok"]:
        raise SystemExit(
            f"HARD regression guard FAILED: {payload['regression_guard']['hard_errors']}"
        )


if __name__ == "__main__":
    main()
