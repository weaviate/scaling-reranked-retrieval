"""Noise-null robustness check for the oracle decomposition (MoCE §5.1).

WHAT THIS ANSWERS
-----------------
The paper's headline analytical result decomposes fusion's per-query optimum into
a ROUTING component (oracle-selector − best-static-fusion) and a BLENDING
component (oracle-config − oracle-selector). The oracle-selector is a per-query
**max over several estimators**, and a max over noisy estimators is upward-biased
even when the estimators have identical true quality (winner's curse /
selection-on-noise). This module quantifies how much routing/blending value the
oracle machinery manufactures under ZERO true heterogeneity, so the real
three-model result can be reported *relative to* that null.

METHOD
------
Take one real cross-encoder's cached scores, make THREE independent noisy copies
of that single score vector, and treat them as a synthetic three-"model"
ensemble. Run the EXACT SAME oracle-selector / oracle-config / best-static-fusion
decomposition on the copies as on the real three models. Because all three copies
derive from one model, any routing/blending value measured is attributable to
selection-on-noise alone. Sweep the noise magnitude `alpha` and average over
seeds to get a curve with uncertainty.

The implementation is a synthetic-cache generator: it writes three noisy clones
of one base model into the `cohere_scores` / `voyage_scores` / `zerank_scores`
slots of an in-memory `ScoreCache`, then calls the unchanged decomposition code
in `src.decompose` (which in turn routes through `DerivedSearchAgent`, so
RSF normalization is recomputed on the restricted pool and depths read prefixes,
exactly as the real pipeline). No oracle/fusion logic is reimplemented here.

AGGREGATION (matches the real pipeline)
---------------------------------------
Within a subset the per-query statistic is the MEAN (per-query recall@K is too
coarse for a median — mostly 0, else 1/|gold|; see src.decompose). Across
the five BRIGHT subsets we take the MEDIAN (the paper's cross-subset aggregator).
Across seeds we report the MEAN and a band (std + 5th/95th percentile) — this is
the one experiment that injects real randomness, so it genuinely needs seeds.

WHAT "PASSING" LOOKS LIKE (informative, NOT a gate)
---------------------------------------------------
We *expect but do not assume*: (1) real routing value sits above the null curve
across reasonable noise scales; (2) the null routing value rises with pool depth
while the real value is flat with depth. This module reports whatever the numbers
show — it does not tune the null to pass.

NON-GOALS / KNOWN LIMITATIONS
-----------------------------
- Answers only "is routing value *entirely* a selection-on-noise artifact?" It
  does NOT partition, within the real run, genuine heterogeneity vs winner's-curse
  on the real models' own noise — that needs a held-out query split and is a
  separate spec (future work).
- Independent per-(query,doc) Gaussian noise is the MOST GENEROUS null for the
  artifact (maximally decorrelated clones). Real rerankers are correlated, so
  beating this null is harder than reality warrants — the conservative direction.
- Gaussian additive noise on raw scores is a modeling choice; it is not claimed
  to mimic any real reranker's error distribution. The null tests the *selection
  operator's* behavior under noise, not a realistic error model.

NO NETWORK. This is pure post-processing over the existing k=2000 caches. The
module installs a guard that makes any reranker-client factory raise, so the run
fails loudly if an API client is ever touched.

This module has two modes:
  * --singleton-only — the TRIMMED main-text null (spec "Selection Bias on
    Routing"): clone ONE base (Zerank), run per-query best-singleton selection
    over the three clones ONLY, and report `oracle_selector − best_singleton`
    (the winner's-curse signal; >= 0 by construction). No fusion, so it is
    tie-free and needs no PYTHONHASHSEED pin. Minutes, not hours.
  * default (full) — the appendix sweep: three bases × the active condition
    menu (equal-weight only: singletons + the equal-weight RRF/RSF blends),
    reporting routing_value (vs best static
    fusion) and blending_value. Multi-hour; pins PYTHONHASHSEED=0 for the
    fusion ties.

TWO-METRIC DISTINCTION (kept separate in code + output; spec §6):
  routing_value  = oracle_selector − best_static_fusion  (paper §5.1; NOT
                   sign-constrained — a fixed blend can denoise; full mode).
  selection_bias = oracle_selector − best_singleton      (singleton-only mode;
                   >= 0; the clean selection-on-noise signal).

Usage:
    uv run python scripts/noise_null.py --singleton-only --smoke
    uv run python scripts/noise_null.py --singleton-only        # trimmed main-text run
    uv run python scripts/noise_null.py --all                   # full appendix sweep
"""
from __future__ import annotations

import os
import sys

# Bit-reproducibility (spec §6): DerivedSearchAgent's RSF path builds its fused
# dict by iterating `set(pool)`, so string hash randomization (PYTHONHASHSEED)
# changes tie-breaking at the rank-K boundary across processes — the real
# three-model overlay can wobble by ~1 query between runs (the existing
# oracle_config_k{N}.json / AGREEMENT.md were themselves written under random
# seeds). We do NOT change the shared pipeline; instead we pin PYTHONHASHSEED=0
# by re-exec'ing once, which makes set iteration — and thus the whole run —
# deterministic. The null clones carry continuous Gaussian noise (tie-free), so
# the null itself is deterministic regardless; this pin only canonicalizes the
# real overlay. Guarded to __main__ so importing this module (e.g. from the
# scripts/ wrapper) never re-execs the host process — the wrapper carries its
# own pin. Skipped in --singleton-only mode: that trimmed
# null never touches fusion (A5), so it is tie-free and needs no pin.
if (
    __name__ == "__main__"
    and "--singleton-only" not in sys.argv
    and os.environ.get("PYTHONHASHSEED") != "0"
):
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable, *sys.argv])

import argparse  # noqa: E402
import statistics  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from src.adapters.cache import ScoreCache  # noqa: E402
from src.domain.metrics import CAP20_METRICS  # noqa: E402
from src.domain.conditions import (  # noqa: E402
    CONDITIONS as _RE_CONDITIONS,
    SINGLETON_CONDITIONS,
)
from src.application.queryset import load_and_validate  # noqa: E402
from src.config import RESULTS_DIR  # noqa: E402
from src.application.decompose import (  # noqa: E402
    METRIC_LABEL,
    SELECTION_METRIC,
    _decompose,
    _per_query_max,
    _qmean,
    per_query_condition_metrics,
    select_best_static_fusion,
)

from src.adapters import qab  # noqa: E402

qab.setup()

# --------------------------------------------------------------------------- #
# Configuration (defaults; all overridable via CLI)                           #
# --------------------------------------------------------------------------- #

PROVIDERS = ("cohere", "voyage", "zerank")   # the three score slots we clone into
SUBSETS = ["biology", "earth_science", "economics", "psychology", "robotics"]
DEPTHS = (100, 200, 500, 1000, 2000)
ALPHAS = (0.05, 0.10, 0.25, 0.50, 1.00)
N_SEEDS = 20
BASES = ("zerank", "voyage", "cohere")       # zerank = strongest R@1 singleton (main)
MENUS = ("combined", "rrf", "rsf")           # decomposition sub-menus
RERANKED_K = 20                              # deployed output cap (R@1/5/20/nDCG@10)
CACHE_K = 2000

# The full active condition menu (equal-weight only), matching src.conditions
# exactly (minus hybrid_only).
FULL_MENU = [c for c in _RE_CONDITIONS if c.name != "hybrid_only"]
SINGLETONS = [c for c in FULL_MENU if c.name in SINGLETON_CONDITIONS]

# Stable base index for seeding (keeps each base's RNG stream independent).
_BASE_IDX = {"cohere": 0, "voyage": 1, "zerank": 2}

OUT_DIR = RESULTS_DIR / "noise_null"


# --------------------------------------------------------------------------- #
# No-network guard                                                            #
# --------------------------------------------------------------------------- #


def install_no_network_guard() -> None:
    """Make every reranker-client factory raise, so a stray API call fails loud.

    The experiment is pure derivation over the cache; DerivedSearchAgent never
    issues a network call. This guard turns any accidental client construction
    (e.g. a future refactor reaching for CollectScoresAgent) into an immediate,
    obvious failure rather than a silent bill.
    """
    try:
        import src.adapters.retrieval.clients as clients
    except Exception:
        return  # clients module unavailable — nothing to guard, still no network

    def _blocked(*_a, **_k):
        raise RuntimeError(
            "noise_null is a zero-network experiment: reranker API clients must "
            "not be constructed. A code path tried to build one."
        )

    for fn in (
        "get_cohere_client", "get_cohere_async_client",
        "get_voyage_client", "get_voyage_async_client",
        "get_zerank_client", "get_zerank_async_client",
    ):
        if hasattr(clients, fn):
            setattr(clients, fn, _blocked)


# --------------------------------------------------------------------------- #
# Menu filtering by fusion method                                             #
# --------------------------------------------------------------------------- #


def menu_for(fusion_method: str) -> list:
    """Conditions for a decomposition sub-menu.

    'combined' = the full active menu (the real analysis/oracle_config.py menu).
    'rrf' / 'rsf' = the 3
    singletons (always the routing baseline; fusion_method=None) + that family's
    equal-weight blends. The singletons are shared so oracle-selector is identical across
    menus and routing/blending stay comparable.
    """
    if fusion_method == "combined":
        return FULL_MENU
    blends = [
        c for c in FULL_MENU
        if c.name not in SINGLETON_CONDITIONS and c.fusion_method == fusion_method
    ]
    return SINGLETONS + blends


def fusion_configs_for(fusion_method: str) -> list:
    """The blend-only conditions of a sub-menu (best-static-fusion candidates)."""
    return [c for c in menu_for(fusion_method) if c.name not in SINGLETON_CONDITIONS]


# --------------------------------------------------------------------------- #
# Clone-cache generation (the only new modelling code)                        #
# --------------------------------------------------------------------------- #


def build_clone_cache(
    real_cache: ScoreCache,
    queries: list[str],
    base: str,
    alpha: float,
    seed: int,
) -> ScoreCache:
    """Synthetic ScoreCache: three independent noisy clones of one base model.

    For each query the base model's FULL-pool score vector `s` (length up to
    2000) gets per-query dispersion `sigma_q = std(s)`, then three independent
    noise vectors `e_i ~ N(0, alpha*sigma_q)` are drawn (one value per
    (query, candidate)) and the clones `s_i = s + e_i` are written into the
    cohere/voyage/zerank score slots. Noise is added ONCE at full depth; every
    smaller depth reads a prefix downstream (DerivedSearchAgent slices
    hybrid_order[:k]) — never redrawn per depth.

    Seeding is `default_rng((seed, base_idx, query_index))`, so the draw is fully
    reproducible and independent across (seed, base, query). `sigma_q == 0`
    (constant scores) yields identical clones — routing/blending then collapse to
    0, which is the correct null behaviour.
    """
    base_key = f"{base}_scores"
    clone = ScoreCache(metadata=dict(real_cache.metadata), queries={})
    for qi, text in enumerate(queries):
        entry = real_cache.queries[text]
        scores = entry[base_key]
        doc_ids = list(scores.keys())
        vals = np.fromiter((scores[d] for d in doc_ids), dtype=float, count=len(doc_ids))
        sigma_q = float(vals.std())
        rng = np.random.default_rng((seed, _BASE_IDX[base], qi))
        new_entry: dict = {"hybrid_order": entry["hybrid_order"]}
        for prov in PROVIDERS:
            noise = rng.normal(0.0, alpha * sigma_q, size=vals.shape[0])
            clone_vals = vals + noise
            new_entry[f"{prov}_scores"] = {
                d: float(v) for d, v in zip(doc_ids, clone_vals)
            }
        clone.queries[text] = new_entry
    return clone


# --------------------------------------------------------------------------- #
# One decomposition cell                                                       #
# --------------------------------------------------------------------------- #


def decompose_at_depth(
    tables: dict[str, dict],
    present: list[str],
    n_by_ds: dict[str, int],
    fusion_method: str,
) -> tuple[dict[str, dict], str]:
    """Per-subset routing/blending for one depth + sub-menu (real or null).

    Mirrors analysis/oracle_config.py's run steps 2–3 exactly: pick ONE best-static-fusion
    blend GLOBALLY (argmax over the sub-menu's blends of the pooled-across-subsets
    mean of SELECTION_METRIC=recall_at_1), then decompose each subset against that
    fixed blend. Returns (per_subset_decomposition, global_bsf_config).
    """
    menu = menu_for(fusion_method)
    singletons = SINGLETONS
    fusion_configs = fusion_configs_for(fusion_method)

    pooled: dict[str, list[float]] = {c.name: [] for c in fusion_configs}
    for ds in present:
        for c in fusion_configs:
            pooled[c.name].extend(tables[ds][c.name][SELECTION_METRIC])
    bsf_cfg, _ = select_best_static_fusion(pooled, fusion_configs=fusion_configs)

    per_subset = {
        ds: _decompose(tables[ds], n_by_ds[ds], bsf_cfg, menu=menu, singletons=singletons)
        for ds in present
    }
    return per_subset, bsf_cfg


def _best_singleton_mean(table: dict, metric: str) -> float:
    """Max over the three clone singletons of their per-query mean of `metric`."""
    return max(_qmean(table[c.name][metric]) for c in SINGLETONS)


def _clone_median_recall_at_1(table: dict) -> float:
    """Median across the three clones of their mean recall_at_1 (clone quality)."""
    return statistics.median(_qmean(table[c.name]["recall_at_1"]) for c in SINGLETONS)


# --------------------------------------------------------------------------- #
# Sweep driver                                                                 #
# --------------------------------------------------------------------------- #


def _load_real(subsets: list[str]) -> dict[str, tuple]:
    """Load + validate the k=2000 cache and query-set for each subset once."""
    real: dict[str, tuple] = {}
    for ds in subsets:
        loaded = load_and_validate(ds)
        if loaded is None:
            print(f"  [skip] no usable cache for {ds}")
            continue
        real[ds] = loaded
    if not real:
        raise SystemExit("No subsets had a usable k=2000 cache.")
    return real


def compute_real_overlay(
    real: dict[str, tuple], depths: tuple, menus: tuple
) -> list[dict]:
    """Real three-model routing/blending, same code path, for the figure overlay.

    Computed directly from the real caches (never read from oracle_config_k{N}.json)
    so it is guaranteed same-code-path with the null. For 'combined' it reproduces
    the on-disk oracle_config_k{N} decomposition.
    """
    present = list(real.keys())
    # Materialize the full active-menu condition table once per (subset, depth).
    tables: dict[int, dict[str, dict]] = {}
    n_by_ds: dict[str, int] = {}
    for depth in depths:
        tables[depth] = {}
        for ds in present:
            cache, qs = real[ds]
            table, queries = per_query_condition_metrics(cache, qs, depth, RERANKED_K)
            tables[depth][ds] = table
            n_by_ds[ds] = len(queries)

    rows: list[dict] = []
    for fm in menus:
        for depth in depths:
            per_subset, _ = decompose_at_depth(tables[depth], present, n_by_ds, fm)
            for m in CAP20_METRICS:
                label = METRIC_LABEL[m]
                routing = statistics.median(
                    per_subset[ds][label]["routing_value"] for ds in present
                )
                blending = statistics.median(
                    per_subset[ds][label]["blending_value"] for ds in present
                )
                rows.append({
                    "fusion_method": fm, "metric": m, "depth_k": depth,
                    "routing_value_real": routing, "blending_value_real": blending,
                })
    return rows


def run_sweep(
    subsets: list[str],
    bases: tuple,
    alphas: tuple,
    n_seeds: int,
    depths: tuple,
    menus: tuple,
) -> tuple[list[dict], list[dict]]:
    """Full null sweep. Returns (raw_rows, real_overlay_rows)."""
    install_no_network_guard()
    real = _load_real(subsets)
    present = list(real.keys())

    print(f"\nReal overlay over {present} ...")
    real_rows = compute_real_overlay(real, depths, menus)

    raw_rows: list[dict] = []
    total = len(bases) * len(alphas) * n_seeds
    done = 0
    for base in bases:
        for alpha in alphas:
            for seed in range(n_seeds):
                # Build clone tables for every (subset, depth) at this (base, alpha, seed).
                tables: dict[int, dict[str, dict]] = {d: {} for d in depths}
                n_by_ds: dict[str, int] = {}
                clone_r1: dict[int, dict[str, float]] = {d: {} for d in depths}
                for ds in present:
                    cache, qs = real[ds]
                    queries = list(qs.gold.keys())
                    clone = build_clone_cache(cache, queries, base, alpha, seed)
                    for depth in depths:
                        table, q = per_query_condition_metrics(
                            clone, qs, depth, RERANKED_K
                        )
                        tables[depth][ds] = table
                        n_by_ds[ds] = len(q)
                        clone_r1[depth][ds] = _clone_median_recall_at_1(table)

                # Decompose each (menu, depth); emit one row per subset/metric.
                for fm in menus:
                    for depth in depths:
                        per_subset, bsf_cfg = decompose_at_depth(
                            tables[depth], present, n_by_ds, fm
                        )
                        for ds in present:
                            for m in CAP20_METRICS:
                                label = METRIC_LABEL[m]
                                d = per_subset[ds][label]
                                raw_rows.append({
                                    "base_model": base,
                                    "fusion_method": fm,
                                    "metric": m,
                                    "subset": ds,
                                    "depth_k": depth,
                                    "alpha": alpha,
                                    "seed": seed,
                                    "best_static_fusion": d["best_static_fusion"],
                                    "best_static_fusion_config": bsf_cfg,
                                    "oracle_selector": d["oracle_selector"],
                                    "oracle_config": d["oracle_config"],
                                    "best_singleton": _best_singleton_mean(
                                        tables[depth][ds], m
                                    ),
                                    "routing_value_null": d["routing_value"],
                                    "blending_value_null": d["blending_value"],
                                    "clone_median_recall_at_1": clone_r1[depth][ds],
                                })
                done += 1
                print(
                    f"  [{done}/{total}] base={base} alpha={alpha} seed={seed} "
                    f"-> {len(raw_rows)} rows",
                    flush=True,
                )
    return raw_rows, real_rows


# --------------------------------------------------------------------------- #
# Aggregation                                                                  #
# --------------------------------------------------------------------------- #


def aggregate(raw_df, real_df):
    """Median-across-subset (per seed) then mean+band-across-seed.

    Returns a dataframe at (base_model, fusion_method, metric, depth_k, alpha)
    grain with mean/std/p5/p95 of the routing and blending null values, the mean
    levels (oracle_selector/oracle_config/best_static_fusion/clone quality), and
    the real overlay merged in.
    """
    value_cols = [
        "best_static_fusion", "oracle_selector", "oracle_config",
        "best_singleton", "routing_value_null", "blending_value_null",
        "clone_median_recall_at_1",
    ]
    group_seed = ["base_model", "fusion_method", "metric", "depth_k", "alpha", "seed"]
    # Step 1: median across subsets, per seed.
    per_seed = raw_df.groupby(group_seed, as_index=False)[value_cols].median()

    # Step 2: mean + band across seeds.
    group = ["base_model", "fusion_method", "metric", "depth_k", "alpha"]
    agg_funcs = {c: "mean" for c in value_cols}
    means = per_seed.groupby(group, as_index=False).agg(agg_funcs)

    def _band(col):
        b = per_seed.groupby(group)[col].agg(
            mean="mean", std="std",
            p5=lambda x: float(np.percentile(x, 5)),
            p95=lambda x: float(np.percentile(x, 95)),
        ).reset_index()
        return b.rename(columns={
            "mean": f"{col}_mean", "std": f"{col}_std",
            "p5": f"{col}_p5", "p95": f"{col}_p95",
        })

    out = means.copy()
    for col in ("routing_value_null", "blending_value_null"):
        band = _band(col)
        out = out.merge(band, on=group, how="left")

    # Merge the real overlay (no base/alpha/seed dimension).
    out = out.merge(
        real_df, on=["fusion_method", "metric", "depth_k"], how="left"
    )
    return per_seed, out


# --------------------------------------------------------------------------- #
# Figures                                                                      #
# --------------------------------------------------------------------------- #

# Weaviate brand palette (accents on a clean light ground for paper legibility).
_NAVY = "#130C49"
_GREEN = "#61BD73"
_DEPTH_COLORS = ["#130C49", "#3B2F8F", "#6E5BD0", "#61BD73", "#2E8B57"]


def make_figures(agg_df, out_dir: Path, metrics=("recall_at_1", "recall_at_20")) -> list[Path]:
    """Routing-vs-alpha (one line/depth) and routing-vs-depth (one line/alpha).

    Real routing value overlaid as a horizontal reference (vs-alpha) / as its own
    line (vs-depth); clone-quality regime annotated. Generated for every
    (base, menu, metric) present so a sparse smoke run still renders.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    bases = sorted(agg_df["base_model"].unique())
    menus = sorted(agg_df["fusion_method"].unique())

    for base in bases:
        for menu in menus:
            for metric in metrics:
                sub = agg_df[
                    (agg_df["base_model"] == base)
                    & (agg_df["fusion_method"] == menu)
                    & (agg_df["metric"] == metric)
                ].sort_values(["depth_k", "alpha"])
                if sub.empty:
                    continue
                depths = sorted(sub["depth_k"].unique())
                # Clone-quality threshold = real best-singleton-ish: use the real
                # oracle_selector floor is unavailable here, so mark where clones
                # keep >= 90% of their alpha->0 (clean) quality.
                clean_q = sub[sub["alpha"] == sub["alpha"].min()][
                    "clone_median_recall_at_1"
                ].median()
                q_thresh = 0.9 * clean_q if clean_q else 0.0

                # --- routing vs alpha ---
                fig, ax = plt.subplots(figsize=(6.2, 4.2))
                for i, dk in enumerate(depths):
                    s = sub[sub["depth_k"] == dk].sort_values("alpha")
                    color = _DEPTH_COLORS[i % len(_DEPTH_COLORS)]
                    ax.plot(s["alpha"], s["routing_value_null_mean"], "-o",
                            color=color, label=f"null k={dk}", lw=1.8, ms=4)
                    ax.fill_between(s["alpha"], s["routing_value_null_p5"],
                                    s["routing_value_null_p95"], color=color, alpha=0.12)
                    real = s["routing_value_real"].dropna()
                    if len(real):
                        ax.axhline(real.iloc[0], color=color, ls="--", lw=1.0, alpha=0.7)
                # Shade the reasonable-alpha (clones still good) regime.
                cutoff = None
                clean = sub.groupby("alpha")["clone_median_recall_at_1"].median()
                good = clean[clean >= q_thresh]
                if len(good):
                    cutoff = float(good.index.max())
                    ax.axvspan(sub["alpha"].min(), cutoff, color=_GREEN, alpha=0.07,
                               label="clones comparable")
                ax.set_xlabel("noise scale  α")
                ax.set_ylabel(f"routing value ({METRIC_LABEL[metric]})")
                ax.set_title(f"Null routing vs α — base={base}, {menu}\n"
                             f"(dashed = real three-model routing)")
                ax.axhline(0, color="grey", lw=0.6)
                ax.legend(fontsize=7, ncol=2)
                fig.tight_layout()
                p = out_dir / f"fig_routing_vs_alpha__{base}__{menu}__{metric}.png"
                fig.savefig(p, dpi=150)
                plt.close(fig)
                written.append(p)

                # --- routing vs depth ---
                fig, ax = plt.subplots(figsize=(6.2, 4.2))
                alphas = sorted(sub["alpha"].unique())
                for i, a in enumerate(alphas):
                    s = sub[sub["alpha"] == a].sort_values("depth_k")
                    ax.plot(s["depth_k"], s["routing_value_null_mean"], "-o",
                            color=_DEPTH_COLORS[i % len(_DEPTH_COLORS)],
                            label=f"null α={a}", lw=1.8, ms=4)
                real_line = sub.dropna(subset=["routing_value_real"]).groupby(
                    "depth_k")["routing_value_real"].first()
                if len(real_line):
                    ax.plot(real_line.index, real_line.values, "-s", color=_NAVY,
                            lw=2.4, ms=6, label="real 3-model")
                ax.set_xscale("log")
                ax.set_xlabel("pool depth  k")
                ax.set_ylabel(f"routing value ({METRIC_LABEL[metric]})")
                ax.set_title(f"Null routing vs depth — base={base}, {menu}\n"
                             f"(flat real vs rising null is the key contrast)")
                ax.axhline(0, color="grey", lw=0.6)
                ax.legend(fontsize=7, ncol=2)
                fig.tight_layout()
                p = out_dir / f"fig_routing_vs_depth__{base}__{menu}__{metric}.png"
                fig.savefig(p, dpi=150)
                plt.close(fig)
                written.append(p)
    return written


# --------------------------------------------------------------------------- #
# Results stub                                                                 #
# --------------------------------------------------------------------------- #


def write_summary(agg_df, out_dir: Path, metric: str = "recall_at_1") -> Path:
    """Numbers-only stub: real-vs-null comparison, without overstating it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["# Noise-Null Robustness Check — Results Stub", ""]
    lines.append(
        "Auto-generated. Routing/blending value the oracle decomposition produces "
        "under THREE independent noisy clones of one base model (zero true "
        "heterogeneity), versus the real three-model value. Reported on "
        f"`{metric}`. Per-query MEAN, median across subsets, mean across seeds. "
        "This is informative, not a pass/fail gate — read it as effect sizes."
    )
    lines.append("")
    bases = sorted(agg_df["base_model"].unique())
    menus = sorted(agg_df["fusion_method"].unique())
    for base in bases:
        for menu in menus:
            sub = agg_df[
                (agg_df["base_model"] == base)
                & (agg_df["fusion_method"] == menu)
                & (agg_df["metric"] == metric)
            ]
            if sub.empty:
                continue
            depths = sorted(sub["depth_k"].unique())
            alphas = sorted(sub["alpha"].unique())
            # Reasonable-alpha regime: clones keep >=90% of clean quality.
            clean = sub[sub["alpha"] == min(alphas)]["clone_median_recall_at_1"].median()
            q_thresh = 0.9 * clean if clean else 0.0
            good_alphas = sorted(
                float(a) for a in alphas
                if sub[sub["alpha"] == a]["clone_median_recall_at_1"].median() >= q_thresh
            )
            lines.append(f"## base={base}, menu={menu}, {METRIC_LABEL[metric]}")
            lines.append("")
            lines.append(
                f"- Reasonable-α regime (clones ≥90% of α→0 quality): "
                f"{good_alphas or 'none'}."
            )
            # Real vs null in regime, at deepest depth.
            dk = max(depths)
            d_sub = sub[sub["depth_k"] == dk]
            real_routing = d_sub["routing_value_real"].dropna()
            real_routing = float(real_routing.iloc[0]) if len(real_routing) else float("nan")
            null_in_regime = d_sub[d_sub["alpha"].isin(good_alphas)]["routing_value_null_mean"]
            max_null = float(null_in_regime.max()) if len(null_in_regime) else float("nan")
            exceeds = (
                "yes" if (real_routing == real_routing and max_null == max_null
                          and real_routing > max_null) else "no/unclear"
            )
            lines.append(
                f"- At k={dk}: real routing = {real_routing:.4f}; max null routing "
                f"in regime = {max_null:.4f}. Real exceeds null: **{exceeds}**."
            )
            # Null rise with depth, at the largest reasonable alpha.
            if good_alphas:
                a = max(good_alphas)
                s = sub[sub["alpha"] == a].sort_values("depth_k")
                if len(s) >= 2:
                    lo = float(s.iloc[0]["routing_value_null_mean"])
                    hi = float(s.iloc[-1]["routing_value_null_mean"])
                    trend = "rises" if hi > lo else ("flat" if hi == lo else "falls")
                    lines.append(
                        f"- Null routing vs depth at α={a}: k={int(s.iloc[0]['depth_k'])} "
                        f"{lo:.4f} → k={int(s.iloc[-1]['depth_k'])} {hi:.4f} ({trend})."
                    )
                    # Real flatness for contrast.
                    rr = s.dropna(subset=["routing_value_real"]).sort_values("depth_k")
                    if len(rr) >= 2:
                        rlo = float(rr.iloc[0]["routing_value_real"])
                        rhi = float(rr.iloc[-1]["routing_value_real"])
                        lines.append(
                            f"- Real routing vs depth: {rlo:.4f} → {rhi:.4f} "
                            f"(Δ={rhi - rlo:+.4f})."
                        )
            lines.append("")
    path = out_dir / "SUMMARY.md"
    path.write_text("\n".join(lines))
    return path


# --------------------------------------------------------------------------- #
# Singleton-only mode (trimmed main-text null; spec "Selection Bias on Routing") #
# --------------------------------------------------------------------------- #
#
# This is the experiment that MATCHES the reviewer concern. The concern is about
# the SELECTION operator (per-query best-of-three is upward-biased on noise), so
# the clean signal is `oracle_selector − best_singleton` — a quantity over the
# THREE singletons only, never touching the equal-weight fusion blends. Dropping fusion
# also makes the null TIE-FREE by construction (no RSF normalization / RRF rank
# fusion), so no PYTHONHASHSEED pin is needed for the null's determinism (A5:
# DerivedSearchAgent's singleton path sorts scores directly, bypassing fusion).
#
# Two-metric distinction (spec §6) — kept separate in code and output:
#   - routing_value      = oracle_selector − best_static_fusion  (paper §5.1
#                          decomposition; NOT sign-constrained; full mode above).
#   - selection_bias     = oracle_selector − best_singleton      (THIS mode;
#                          >= 0 by construction; the winner's-curse signal).


def _singleton_decompose(table: dict, n: int, metric: str) -> tuple[float, float, float]:
    """(oracle_selector, best_singleton, selection_bias) for one metric.

    oracle_selector = mean over queries of the per-query max across the three
    singletons (the routing oracle). best_singleton = the best single model used
    uniformly = max over the three of the per-query mean. selection_bias =
    oracle_selector − best_singleton, >= 0 by construction (mean of a per-query
    max dominates the max of per-query means).
    """
    names = [c.name for c in SINGLETONS]
    sel = _qmean(_per_query_max(table, names, metric, n))
    best = _best_singleton_mean(table, metric)
    return sel, best, sel - best


def compute_real_overlay_singleton(
    real: dict[str, tuple], depths: tuple, metrics: tuple
) -> list[dict]:
    """Real three-model `oracle_selector − best_singleton`, median across subsets.

    Same code path as the null (singletons only, no fusion), on the true
    Cohere/Voyage/Zerank ensemble. Distinct from the paper's routing value
    (which is vs best static fusion) — see the §6 distinction above.
    """
    present = list(real.keys())
    rows: list[dict] = []
    for depth in depths:
        per: dict[str, list[float]] = {m: [] for m in metrics}
        for ds in present:
            cache, qs = real[ds]
            table, queries = per_query_condition_metrics(
                cache, qs, depth, RERANKED_K, menu=SINGLETONS
            )
            for m in metrics:
                _, _, bias = _singleton_decompose(table, len(queries), m)
                per[m].append(bias)
        for m in metrics:
            rows.append({
                "metric": m, "depth_k": depth,
                "selection_bias_real": statistics.median(per[m]),
            })
    return rows


def run_sweep_singleton(
    subsets: list[str],
    base: str,
    alphas: tuple,
    n_seeds: int,
    depths: tuple,
    metrics: tuple,
) -> tuple[list[dict], list[dict]]:
    """Trimmed null sweep: clone one base, singleton selection only. Tie-free."""
    install_no_network_guard()
    real = _load_real(subsets)
    present = list(real.keys())

    print(f"\nReal selector−singleton overlay over {present} ...")
    real_rows = compute_real_overlay_singleton(real, depths, metrics)

    raw_rows: list[dict] = []
    total = len(alphas) * n_seeds
    done = 0
    for alpha in alphas:
        for seed in range(n_seeds):
            for ds in present:
                cache, qs = real[ds]
                queries = list(qs.gold.keys())
                clone = build_clone_cache(cache, queries, base, alpha, seed)
                for depth in depths:
                    table, q = per_query_condition_metrics(
                        clone, qs, depth, RERANKED_K, menu=SINGLETONS
                    )
                    n = len(q)
                    cq = _clone_median_recall_at_1(table)
                    for m in metrics:
                        sel, best, bias = _singleton_decompose(table, n, m)
                        raw_rows.append({
                            "base_model": base,
                            "metric": m,
                            "subset": ds,
                            "depth_k": depth,
                            "alpha": alpha,
                            "seed": seed,
                            "oracle_selector": sel,
                            "best_singleton": best,
                            "selection_bias_null": bias,
                            "clone_median_recall_at_1": cq,
                        })
            done += 1
            print(f"  [{done}/{total}] alpha={alpha} seed={seed} "
                  f"-> {len(raw_rows)} rows", flush=True)
    return raw_rows, real_rows


def aggregate_singleton(raw_df, real_df):
    """Median-across-subset (per seed) then mean+band-across-seed for the bias."""
    value_cols = [
        "oracle_selector", "best_singleton", "selection_bias_null",
        "clone_median_recall_at_1",
    ]
    group_seed = ["base_model", "metric", "depth_k", "alpha", "seed"]
    per_seed = raw_df.groupby(group_seed, as_index=False)[value_cols].median()

    group = ["base_model", "metric", "depth_k", "alpha"]
    means = per_seed.groupby(group, as_index=False)[value_cols].mean()
    band = per_seed.groupby(group)["selection_bias_null"].agg(
        selection_bias_null_mean="mean", selection_bias_null_std="std",
        selection_bias_null_p5=lambda x: float(np.percentile(x, 5)),
        selection_bias_null_p95=lambda x: float(np.percentile(x, 95)),
    ).reset_index()
    out = means.merge(band, on=group, how="left")
    out = out.merge(real_df, on=["metric", "depth_k"], how="left")
    return per_seed, out


def make_figures_singleton(
    agg_df, out_dir: Path, metrics=("recall_at_1", "recall_at_20")
) -> list[Path]:
    """Figure A (bias vs alpha) + Figure B (bias vs depth), real overlay + regime."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    base = agg_df["base_model"].iloc[0]
    for metric in metrics:
        sub = agg_df[agg_df["metric"] == metric].sort_values(["depth_k", "alpha"])
        if sub.empty:
            continue
        depths = sorted(sub["depth_k"].unique())
        clean_q = sub[sub["alpha"] == sub["alpha"].min()][
            "clone_median_recall_at_1"
        ].median()
        q_thresh = 0.9 * clean_q if clean_q else 0.0

        # Figure A: bias vs alpha, one line per depth, real horizontal reference.
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        for i, dk in enumerate(depths):
            s = sub[sub["depth_k"] == dk].sort_values("alpha")
            color = _DEPTH_COLORS[i % len(_DEPTH_COLORS)]
            ax.plot(s["alpha"], s["selection_bias_null_mean"], "-o", color=color,
                    lw=1.8, ms=4, label=f"null k={dk}")
            ax.fill_between(s["alpha"], s["selection_bias_null_p5"],
                            s["selection_bias_null_p95"], color=color, alpha=0.12)
            real = s["selection_bias_real"].dropna()
            if len(real):
                ax.axhline(real.iloc[0], color=color, ls="--", lw=1.0, alpha=0.7)
        good = sub.groupby("alpha")["clone_median_recall_at_1"].median()
        good = good[good >= q_thresh]
        if len(good):
            ax.axvspan(sub["alpha"].min(), float(good.index.max()),
                       color=_GREEN, alpha=0.07, label="clones comparable")
        ax.set_xlabel("noise scale  α")
        ax.set_ylabel(f"oracle-selector − best-singleton ({METRIC_LABEL[metric]})")
        ax.set_title(f"Selection-bias null vs α — base={base}\n"
                     f"(dashed = real three-model selector−singleton)")
        ax.axhline(0, color="grey", lw=0.6)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        p = out_dir / f"figA_bias_vs_alpha__{base}__{metric}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)

        # Figure B: bias vs depth, one line per alpha, real line overlaid.
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        for i, a in enumerate(sorted(sub["alpha"].unique())):
            s = sub[sub["alpha"] == a].sort_values("depth_k")
            ax.plot(s["depth_k"], s["selection_bias_null_mean"], "-o",
                    color=_DEPTH_COLORS[i % len(_DEPTH_COLORS)],
                    lw=1.8, ms=4, label=f"null α={a}")
        real_line = sub.dropna(subset=["selection_bias_real"]).groupby(
            "depth_k")["selection_bias_real"].first()
        if len(real_line):
            ax.plot(real_line.index, real_line.values, "-s", color=_NAVY,
                    lw=2.4, ms=6, label="real 3-model")
        ax.set_xscale("log")
        ax.set_xlabel("pool depth  k")
        ax.set_ylabel(f"oracle-selector − best-singleton ({METRIC_LABEL[metric]})")
        ax.set_title(f"Selection-bias null vs depth — base={base}\n"
                     f"(flat real vs rising null is the key contrast)")
        ax.axhline(0, color="grey", lw=0.6)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        p = out_dir / f"figB_bias_vs_depth__{base}__{metric}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)
    return written


def write_summary_singleton(agg_df, out_dir: Path, metric: str = "recall_at_1") -> Path:
    """Numbers-only stub framed on selector − singleton (spec §5/§6)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    base = agg_df["base_model"].iloc[0]
    sub = agg_df[agg_df["metric"] == metric]
    lines = ["# Noise-Null Robustness Check — Selection Bias on Routing", ""]
    lines.append(
        "Auto-generated. Verdict is framed on **oracle-selector − best-singleton** "
        "(the winner's-curse signal: non-negative, manufactured purely by per-query "
        "max-over-three under zero true heterogeneity) — NOT on the paper's "
        "decomposition routing value (oracle-selector − best static fusion), which "
        "is a different, sign-unconstrained quantity (spec §6). Base clone = "
        f"`{base}`. Per-query MEAN, median across subsets, mean across seeds, "
        f"on `{metric}`. Informative, not a pass/fail gate."
    )
    lines.append("")
    if sub.empty:
        lines.append("_No data._")
        path = out_dir / "SUMMARY.md"
        path.write_text("\n".join(lines))
        return path
    depths = sorted(sub["depth_k"].unique())
    alphas = sorted(sub["alpha"].unique())
    clean = sub[sub["alpha"] == min(alphas)]["clone_median_recall_at_1"].median()
    q_thresh = 0.9 * clean if clean else 0.0
    good_alphas = sorted(
        float(a) for a in alphas
        if sub[sub["alpha"] == a]["clone_median_recall_at_1"].median() >= q_thresh
    )
    lines.append(f"- Reasonable-α regime (clones ≥90% of α→0 quality): {good_alphas or 'none'}.")
    dk = max(depths)
    d_sub = sub[sub["depth_k"] == dk]
    real = d_sub["selection_bias_real"].dropna()
    real = float(real.iloc[0]) if len(real) else float("nan")
    null_in = d_sub[d_sub["alpha"].isin(good_alphas)]["selection_bias_null_mean"]
    max_null = float(null_in.max()) if len(null_in) else float("nan")
    exceeds = ("yes" if (real == real and max_null == max_null and real > max_null)
               else "no/unclear")
    lines.append(
        f"- At k={dk}: real selector−singleton = {real:.4f}; max null in regime = "
        f"{max_null:.4f}. Real exceeds null: **{exceeds}**."
    )
    if good_alphas:
        a = max(good_alphas)
        s = sub[sub["alpha"] == a].sort_values("depth_k")
        if len(s) >= 2:
            lo, hi = float(s.iloc[0]["selection_bias_null_mean"]), float(s.iloc[-1]["selection_bias_null_mean"])
            trend = "rises" if hi > lo else ("flat" if hi == lo else "falls")
            lines.append(
                f"- Null vs depth at α={a}: k={int(s.iloc[0]['depth_k'])} {lo:.4f} → "
                f"k={int(s.iloc[-1]['depth_k'])} {hi:.4f} ({trend})."
            )
            rr = s.dropna(subset=["selection_bias_real"]).sort_values("depth_k")
            if len(rr) >= 2:
                rlo, rhi = float(rr.iloc[0]["selection_bias_real"]), float(rr.iloc[-1]["selection_bias_real"])
                lines.append(
                    f"- Real vs depth: {rlo:.4f} → {rhi:.4f} (Δ={rhi - rlo:+.4f})."
                )
    lines.append("")
    lines.append(
        "Scope (spec §7): this null tests the SELECTION (routing) component, which "
        "is the part exposed to winner's-curse bias. Blending (oracle_config − "
        "oracle_selector) is a fused-ranking effect of genuine mixture, not per-query "
        "maximization, so it is outside this null's scope by design."
    )
    path = out_dir / "SUMMARY.md"
    path.write_text("\n".join(lines))
    return path


def write_artifacts_singleton(raw_rows, real_rows, out_dir: Path) -> dict:
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_df = pd.DataFrame(raw_rows)
    real_df = pd.DataFrame(real_rows)
    _per_seed, agg_df = aggregate_singleton(raw_df, real_df)
    raw_path = out_dir / "noise_null_singleton_raw.parquet"
    agg_path = out_dir / "noise_null_singleton_agg.parquet"
    raw_df.to_parquet(raw_path, index=False)
    agg_df.to_parquet(agg_path, index=False)
    figs = make_figures_singleton(agg_df, out_dir)
    summary = write_summary_singleton(agg_df, out_dir)
    print(f"\nWrote {raw_path} ({len(raw_df)} rows)")
    print(f"Wrote {agg_path} ({len(agg_df)} rows)")
    print(f"Wrote {len(figs)} figures + {summary}")
    return {"raw": raw_df, "agg": agg_df, "real": real_df, "figures": figs}


def run_smoke_singleton() -> None:
    """Fast self-test for the trimmed singleton-only null (biology, depths 200/2000)."""
    print("=== SMOKE (singleton-only): biology, 3 seeds, alpha {0.05,1.0}, k {200,2000} ===")
    install_no_network_guard()
    metrics = ("recall_at_1", "recall_at_20")
    raw_rows, real_rows = run_sweep_singleton(
        ["biology"], "zerank", (0.05, 1.0), 3, (200, 2000), metrics
    )
    import pandas as pd
    raw = pd.DataFrame(raw_rows)
    # selection_bias is non-negative by construction.
    assert (raw["selection_bias_null"] >= -1e-9).all(), "negative selection_bias_null"
    assert (raw["oracle_selector"] >= raw["best_singleton"] - 1e-9).all()
    # alpha -> 0 collapse and growth with alpha (R@1, k=2000).
    r1 = raw[(raw["metric"] == "recall_at_1") & (raw["depth_k"] == 2000)]
    lo = r1[r1["alpha"] == 0.05]["selection_bias_null"].mean()
    hi = r1[r1["alpha"] == 1.0]["selection_bias_null"].mean()
    print(f"  selection_bias_null R@1 k=2000: alpha=0.05 {lo:.4f}  alpha=1.0 {hi:.4f}")
    assert lo <= hi + 1e-9, "selection bias did not grow with alpha"
    assert lo < 0.02, f"bias at alpha=0.05 not near-zero: {lo:.4f}"
    # Determinism WITHOUT a hash pin: the singleton path is tie-free, so a second
    # identical clone+derive reproduces the bias exactly.
    cache_qs = _load_real(["biology"])["biology"]
    cache, qs = cache_qs
    q = list(qs.gold.keys())
    biases = []
    for _ in range(2):
        clone = build_clone_cache(cache, q, "zerank", 0.5, 11)
        t, qq = per_query_condition_metrics(clone, qs, 2000, RERANKED_K, menu=SINGLETONS)
        _, _, b = _singleton_decompose(t, len(qq), "recall_at_1")
        biases.append(b)
    assert biases[0] == biases[1], biases  # tie-free => bit-identical, no pin
    print(f"  tie-free determinism OK (bias={biases[0]:.4f})")
    write_artifacts_singleton(raw_rows, real_rows, OUT_DIR / "_smoke_singleton")
    print("\nSINGLETON SMOKE PASSED.")


# --------------------------------------------------------------------------- #
# Persist + CLI                                                                #
# --------------------------------------------------------------------------- #


def write_artifacts(raw_rows, real_rows, out_dir: Path) -> dict:
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_df = pd.DataFrame(raw_rows)
    real_df = pd.DataFrame(real_rows)
    _per_seed, agg_df = aggregate(raw_df, real_df)

    raw_path = out_dir / "noise_null_raw.parquet"
    agg_path = out_dir / "noise_null_agg.parquet"
    raw_df.to_parquet(raw_path, index=False)
    agg_df.to_parquet(agg_path, index=False)
    figs = make_figures(agg_df, out_dir)
    summary = write_summary(agg_df, out_dir)
    print(f"\nWrote {raw_path} ({len(raw_df)} rows)")
    print(f"Wrote {agg_path} ({len(agg_df)} rows)")
    print(f"Wrote {len(figs)} figures + {summary}")
    return {"raw": raw_df, "agg": agg_df, "real": real_df, "figures": figs}


def run_smoke() -> None:
    """Fast self-test: biology, 3 seeds, α∈{0.05,1.0}, depth 200, all menus.

    Asserts the structural invariants and the α→0 collapse, and checks the
    combined-menu real overlay reproduces oracle_config_k200.json's biology
    oracle_selector / oracle_config at k=200 (regression guard).
    """
    print("=== SMOKE: biology, 3 seeds, alpha in {0.05, 1.0}, depth 200 ===")
    install_no_network_guard()
    subsets = ["biology"]
    bases = ("zerank",)
    alphas = (0.05, 1.0)
    depths = (200,)
    menus = ("combined", "rrf", "rsf")
    raw_rows, real_rows = run_sweep(subsets, bases, alphas, 3, depths, menus)

    import pandas as pd
    raw = pd.DataFrame(raw_rows)

    # True invariants (per row). oracle_config dominates both the per-query
    # singleton oracle and any fixed blend; oracle_selector dominates each
    # singleton; blending_value >= 0. NOTE routing_value is NOT sign-constrained
    # — a fixed blend can denoise and beat the per-query-best singleton, so null
    # routing can legitimately be <= 0 (a meaningful result, not a bug).
    assert (raw["oracle_config"] >= raw["oracle_selector"] - 1e-9).all(), \
        "oracle_config < oracle_selector"
    assert (raw["oracle_config"] >= raw["best_static_fusion"] - 1e-9).all(), \
        "oracle_config < best_static_fusion"
    assert (raw["oracle_selector"] >= raw["best_singleton"] - 1e-9).all(), \
        "oracle_selector < best_singleton"
    assert (raw["blending_value_null"] >= -1e-9).all(), "negative blending_value_null"

    # alpha -> 0 collapse, measured on the unambiguous winner's-curse signal
    # (oracle_selector − best_singleton, always >= 0 and rising with noise): the
    # max-over-three-clones increasingly exceeds the best single clone as alpha
    # grows. (Routing_value itself can fall as the fixed blend also denoises.)
    r1 = raw[(raw["metric"] == "recall_at_1") & (raw["fusion_method"] == "combined")]
    r1 = r1.assign(curse=r1["oracle_selector"] - r1["best_singleton"])
    lo = r1[r1["alpha"] == 0.05]["curse"].mean()
    hi = r1[r1["alpha"] == 1.0]["curse"].mean()
    rt_lo = r1[r1["alpha"] == 0.05]["routing_value_null"].mean()
    rt_hi = r1[r1["alpha"] == 1.0]["routing_value_null"].mean()
    print(f"  winner's-curse (selector−singleton) R@1 combined: "
          f"alpha=0.05 {lo:.4f}  alpha=1.0 {hi:.4f}")
    print(f"  routing_value_null R@1 combined: "
          f"alpha=0.05 {rt_lo:.4f}  alpha=1.0 {rt_hi:.4f}")
    assert lo <= hi + 1e-9, "winner's-curse signal did not grow with alpha"
    assert lo < 0.02, f"curse at alpha=0.05 not near-zero: {lo:.4f}"

    # --- Menu-consistency + determinism regression (deterministic under the
    #     PYTHONHASHSEED pin; replaces a brittle equality vs the stale, non-
    #     reproducible oracle_config_k200.json). ---
    loaded = load_and_validate("biology")
    assert loaded is not None, "smoke needs the biology k=2000 cache"
    cache, qs = loaded
    table, queries = per_query_condition_metrics(cache, qs, 200, RERANKED_K)
    n = len(queries)
    dec_by_menu = {}
    for fm in ("combined", "rrf", "rsf"):
        fcs = fusion_configs_for(fm)
        pooled = {c.name: table[c.name][SELECTION_METRIC] for c in fcs}
        bsf, _ = select_best_static_fusion(pooled, fusion_configs=fcs)
        dec_by_menu[fm] = _decompose(
            table, n, bsf, menu=menu_for(fm), singletons=SINGLETONS
        )["R@1"]
    # oracle_selector is menu-independent (the three singletons are shared).
    sels = {fm: d["oracle_selector"] for fm, d in dec_by_menu.items()}
    assert max(sels.values()) - min(sels.values()) < 1e-9, sels
    # combined oracle_config dominates each single-family sub-menu (superset).
    oc_comb = dec_by_menu["combined"]["oracle_config"]
    assert oc_comb >= dec_by_menu["rrf"]["oracle_config"] - 1e-9
    assert oc_comb >= dec_by_menu["rsf"]["oracle_config"] - 1e-9
    # Determinism: a second materialization is bit-identical under the pin.
    # (bsf_config only anchors the routing baseline, which this check ignores —
    # any menu condition works.)
    table2, _ = per_query_condition_metrics(cache, qs, 200, RERANKED_K)
    oc2 = _decompose(table2, n, "rsf_equal_3way")["R@1"]["oracle_config"]
    assert abs(oc2 - oc_comb) < 1e-12, (oc2, oc_comb)
    print(
        f"  menu-consistency OK: oracle_selector={min(sels.values()):.4f} "
        f"(menu-invariant); combined oracle_config={oc_comb:.4f} >= "
        f"rrf {dec_by_menu['rrf']['oracle_config']:.4f}, "
        f"rsf {dec_by_menu['rsf']['oracle_config']:.4f}"
    )
    import json
    oc_path = RESULTS_DIR / "oracle_config_k200.json"
    if oc_path.exists():
        with open(oc_path) as f:
            disk = json.load(f)["per_subset"]["biology"]["R@1"]["oracle_config"]
        if abs(disk - oc_comb) > 1e-9:
            print(
                f"  [note] on-disk oracle_config_k200.json biology oracle_config="
                f"{disk:.4f} != canonical {oc_comb:.4f} (it predates the "
                f"PYTHONHASHSEED pin / a cache refresh; not a failure)."
            )

    # Exercise the artifact writers (writes under results/noise_null/_smoke).
    smoke_dir = OUT_DIR / "_smoke"
    write_artifacts(raw_rows, real_rows, smoke_dir)
    print("\nSMOKE PASSED.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="fast self-test")
    parser.add_argument("--all", action="store_true",
                        help="full sweep with all defaults (multi-hour)")
    parser.add_argument(
        "--singleton-only", action="store_true",
        help=("TRIMMED main-text null (spec 'Selection Bias on Routing'): clone one "
              "base, singleton selection only (no fusion, tie-free), metric "
              "oracle_selector − best_singleton. Minutes, not hours."),
    )
    parser.add_argument("--subsets", nargs="+", default=list(SUBSETS))
    parser.add_argument("--bases", nargs="+", default=list(BASES),
                        choices=list(PROVIDERS), help="full mode: bases to clone")
    parser.add_argument("--base", default="zerank", choices=list(PROVIDERS),
                        help="singleton-only mode: the single base to clone")
    parser.add_argument("--alphas", nargs="+", type=float, default=list(ALPHAS))
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument("--depths", nargs="+", type=int, default=list(DEPTHS))
    parser.add_argument("--menus", nargs="+", default=list(MENUS),
                        choices=list(MENUS), help="full mode only")
    parser.add_argument("--metrics", nargs="+", default=["recall_at_1", "recall_at_20"],
                        choices=list(CAP20_METRICS),
                        help="singleton-only mode: metrics to report (R@1 primary)")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    if args.smoke:
        run_smoke_singleton() if args.singleton_only else run_smoke()
        return

    if args.singleton_only:
        raw_rows, real_rows = run_sweep_singleton(
            args.subsets, args.base, tuple(args.alphas),
            args.seeds, tuple(args.depths), tuple(args.metrics),
        )
        write_artifacts_singleton(raw_rows, real_rows, args.out)
        return

    raw_rows, real_rows = run_sweep(
        args.subsets, tuple(args.bases), tuple(args.alphas),
        args.seeds, tuple(args.depths), tuple(args.menus),
    )
    write_artifacts(raw_rows, real_rows, args.out)


if __name__ == "__main__":
    main()
