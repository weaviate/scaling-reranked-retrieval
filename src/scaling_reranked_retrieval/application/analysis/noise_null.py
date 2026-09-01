"""Noise-null robustness check for the oracle decomposition.

Runs the unchanged decomposition on three independent noisy clones of one base
model, so any routing/blending value measured is selection-on-noise alone.
Aggregation matches the real pipeline: per-query MEAN, MEDIAN across subsets,
MEAN + band across seeds. Zero network.
"""
from __future__ import annotations

import os
import sys

# Bit-reproducibility: the RSF path iterates set(pool), so PYTHONHASHSEED
# changes tie-breaking at the rank-K boundary. Pin via a one-time re-exec,
# guarded to __main__ so imports never re-exec the host. --singleton-only never
# touches fusion, is tie-free, and needs no pin.
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

from scaling_reranked_retrieval.adapters.cache import ScoreCache  # noqa: E402
from scaling_reranked_retrieval.domain.metrics import CAP20_METRICS  # noqa: E402
from scaling_reranked_retrieval.domain.conditions import (  # noqa: E402
    CONDITIONS as _RE_CONDITIONS,
    SINGLETON_CONDITIONS,
)
from scaling_reranked_retrieval.application.queryset import load_and_validate  # noqa: E402
from scaling_reranked_retrieval.config import RESULTS_DIR  # noqa: E402
from scaling_reranked_retrieval.application.decompose import (  # noqa: E402
    METRIC_LABEL,
    SELECTION_METRIC,
    _decompose,
    _per_query_max,
    _qmean,
    per_query_condition_metrics,
    select_best_static_fusion,
)

from scaling_reranked_retrieval.adapters import qab  # noqa: E402

qab.setup()

PROVIDERS = ("cohere", "voyage", "zerank")   # the three score slots we clone into
SUBSETS = ["biology", "earth_science", "economics", "psychology", "robotics"]
DEPTHS = (100, 200, 500, 1000, 2000)
ALPHAS = (0.05, 0.10, 0.25, 0.50, 1.00)
N_SEEDS = 20
BASES = ("zerank", "voyage", "cohere")       # zerank = strongest R@1 singleton
MENUS = ("combined", "rrf", "rsf")           # decomposition sub-menus
RERANKED_K = 20                              # deployed output cap
CACHE_K = 2000

# Active equal-weight condition menu, matching src.conditions minus hybrid_only.
FULL_MENU = [c for c in _RE_CONDITIONS if c.name != "hybrid_only"]
SINGLETONS = [c for c in FULL_MENU if c.name in SINGLETON_CONDITIONS]

# Stable base index for seeding (keeps each base's RNG stream independent).
_BASE_IDX = {"cohere": 0, "voyage": 1, "zerank": 2}

OUT_DIR = RESULTS_DIR / "noise_null"


def install_no_network_guard() -> None:
    """Make every reranker-client factory raise, so a stray API call fails loud."""
    try:
        import scaling_reranked_retrieval.adapters.retrieval.clients as clients
    except Exception:
        return  # nothing to guard

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


def menu_for(fusion_method: str) -> list:
    """Conditions for a decomposition sub-menu; singletons are shared so
    oracle-selector is identical across menus."""
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


def build_clone_cache(
    real_cache: ScoreCache,
    queries: list[str],
    base: str,
    alpha: float,
    seed: int,
) -> ScoreCache:
    """Synthetic ScoreCache: three independent noisy clones of one base model.

    Noise e_i ~ N(0, alpha*std(s)) is added ONCE at full depth; smaller depths
    read prefixes downstream — never redrawn per depth. Seeding
    default_rng((seed, base_idx, query_index)) keeps draws independent and
    reproducible across (seed, base, query).
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


def decompose_at_depth(
    tables: dict[str, dict],
    present: list[str],
    n_by_ds: dict[str, int],
    fusion_method: str,
) -> tuple[dict[str, dict], str]:
    """Per-subset routing/blending for one depth + sub-menu (real or null).

    One best-static-fusion blend is picked GLOBALLY (pooled-across-subsets
    SELECTION_METRIC), then each subset decomposes against that fixed blend.
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
    """Real three-model routing/blending, computed from the real caches so it is
    guaranteed same-code-path with the null."""
    present = list(real.keys())
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


def aggregate(raw_df, real_df):
    """Median-across-subset (per seed) then mean+band-across-seed."""
    value_cols = [
        "best_static_fusion", "oracle_selector", "oracle_config",
        "best_singleton", "routing_value_null", "blending_value_null",
        "clone_median_recall_at_1",
    ]
    group_seed = ["base_model", "fusion_method", "metric", "depth_k", "alpha", "seed"]
    per_seed = raw_df.groupby(group_seed, as_index=False)[value_cols].median()

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

    out = out.merge(
        real_df, on=["fusion_method", "metric", "depth_k"], how="left"
    )
    return per_seed, out


# Weaviate brand palette.
_NAVY = "#130C49"
_GREEN = "#61BD73"
_DEPTH_COLORS = ["#130C49", "#3B2F8F", "#6E5BD0", "#61BD73", "#2E8B57"]


def make_figures(agg_df, out_dir: Path, metrics=("recall_at_1", "recall_at_20")) -> list[Path]:
    """Routing-vs-alpha (one line/depth) and routing-vs-depth (one line/alpha)."""
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
                # Regime marker: clones keep >= 90% of their alpha->0 quality.
                clean_q = sub[sub["alpha"] == sub["alpha"].min()][
                    "clone_median_recall_at_1"
                ].median()
                q_thresh = 0.9 * clean_q if clean_q else 0.0

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


# Singleton-only mode is tie-free (no fusion), so it needs no PYTHONHASHSEED pin.
# Two-metric distinction, kept separate in code and output:
#   routing_value  = oracle_selector − best_static_fusion (NOT sign-constrained; full mode)
#   selection_bias = oracle_selector − best_singleton     (>= 0; this mode)


def _singleton_decompose(table: dict, n: int, metric: str) -> tuple[float, float, float]:
    """(oracle_selector, best_singleton, selection_bias) for one metric.

    selection_bias >= 0 by construction: the mean of a per-query max dominates
    the max of per-query means.
    """
    names = [c.name for c in SINGLETONS]
    sel = _qmean(_per_query_max(table, names, metric, n))
    best = _best_singleton_mean(table, metric)
    return sel, best, sel - best


def compute_real_overlay_singleton(
    real: dict[str, tuple], depths: tuple, metrics: tuple
) -> list[dict]:
    """Real three-model `oracle_selector − best_singleton`, median across subsets.

    Distinct from routing value (which is vs best static fusion) — see above.
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
    """Numbers-only stub framed on selector − singleton."""
    out_dir.mkdir(parents=True, exist_ok=True)
    base = agg_df["base_model"].iloc[0]
    sub = agg_df[agg_df["metric"] == metric]
    lines = ["# Noise-Null Robustness Check — Selection Bias on Routing", ""]
    lines.append(
        "Auto-generated. Verdict is framed on **oracle-selector − best-singleton** "
        "(the winner's-curse signal: non-negative, manufactured purely by per-query "
        "max-over-three under zero true heterogeneity) — NOT on the paper's "
        "decomposition routing value (oracle-selector − best static fusion), which "
        "is a different, sign-unconstrained quantity (spec Section 6). Base clone = "
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
        "Scope (spec Section 7): this null tests the SELECTION (routing) component, which "
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
    assert (raw["selection_bias_null"] >= -1e-9).all(), "negative selection_bias_null"
    assert (raw["oracle_selector"] >= raw["best_singleton"] - 1e-9).all()
    r1 = raw[(raw["metric"] == "recall_at_1") & (raw["depth_k"] == 2000)]
    lo = r1[r1["alpha"] == 0.05]["selection_bias_null"].mean()
    hi = r1[r1["alpha"] == 1.0]["selection_bias_null"].mean()
    print(f"  selection_bias_null R@1 k=2000: alpha=0.05 {lo:.4f}  alpha=1.0 {hi:.4f}")
    assert lo <= hi + 1e-9, "selection bias did not grow with alpha"
    assert lo < 0.02, f"bias at alpha=0.05 not near-zero: {lo:.4f}"
    # Tie-free singleton path: bit-identical without a hash pin.
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
    """Fast self-test: biology, 3 seeds, α∈{0.05,1.0}, depth 200, all menus."""
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

    # routing_value is NOT sign-constrained — a fixed blend can denoise and beat
    # the per-query-best singleton, so null routing can legitimately be <= 0.
    assert (raw["oracle_config"] >= raw["oracle_selector"] - 1e-9).all(), \
        "oracle_config < oracle_selector"
    assert (raw["oracle_config"] >= raw["best_static_fusion"] - 1e-9).all(), \
        "oracle_config < best_static_fusion"
    assert (raw["oracle_selector"] >= raw["best_singleton"] - 1e-9).all(), \
        "oracle_selector < best_singleton"
    assert (raw["blending_value_null"] >= -1e-9).all(), "negative blending_value_null"

    # α→0 collapse measured on the always-nonnegative winner's-curse signal
    # (routing_value itself can fall as the fixed blend also denoises).
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

    # Menu-consistency + determinism regression (deterministic under the
    # PYTHONHASHSEED pin).
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
