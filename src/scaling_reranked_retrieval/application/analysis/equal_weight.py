"""Equal-weight 3-way vs. equal-weight pairs — read-only extraction.

Reads only run-summary JSONs already on disk; never opens caches/ and makes
zero reranker API calls.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import pandas as pd

from scaling_reranked_retrieval.config import RESULTS_DIR

OUT_DIR = RESULTS_DIR / "equal_weight"

ALL_SUBSETS = ["biology", "earth_science", "economics", "psychology", "robotics"]
ALL_KS = [100, 200, 500, 1000, 2000]

FUSIONS = ["rrf", "rsf"]
FAMILY_KEYS = {
    "cv_pair": "{f}_cv_equal",     # cohere + voyage
    "cz_pair": "{f}_cz_equal",     # cohere + zerank
    "vz_pair": "{f}_vz_equal",     # voyage + zerank
    "threeway": "{f}_equal_3way",  # all three
}

# Singletons are fusion-independent; the same value lands on both rrf and rsf rows.
SINGLETON_KEYS = {
    "cohere": "cohere_only",
    "voyage": "voyage_only",
    "zerank": "zerank_only",
}


def equal_weight_keys() -> list[str]:
    """All 8 keys that must be present in every run file we read."""
    return [tmpl.format(f=f) for f in FUSIONS for tmpl in FAMILY_KEYS.values()]


# rk20 metrics come from runs/ (fallback runs_rk100/); rk100 always runs_rk100/.
RK20_METRICS = {
    "recall@1": "recall_at_1",
    "recall@5": "recall_at_5",
    "recall@20": "recall_at_20",
    "ndcg@10": "nDCG_at_10",
}
RK100_METRICS = {
    "recall@50": "recall_at_50",
    "recall@100": "recall_at_100",
}
ALL_METRICS = {**RK20_METRICS, **RK100_METRICS}

METRIC_ORDER = ["recall@1", "recall@20", "recall@5", "ndcg@10", "recall@50", "recall@100"]

# Noise band: no winner declared inside ±0.01 (~1 query on R@1); also covers the
# RSF-equal ~1-query PYTHONHASHSEED tie wobble.
NOISE_BAND = 0.01

# Regression guard vs published cross-domain best-fusion winners; tolerance
# ±0.01 (RSF tie wobble). Catches metric-key mismatches and wrong-file reads.
REGRESSION_GUARD = [
    ("earth_science", 2000, "recall@1", "rsf_equal_3way", 0.579),
    ("earth_science", 200, "recall@1", "rsf_equal_3way", 0.535),
    ("economics", 2000, "recall@20", "rsf_cz_equal", 0.489),
    ("robotics", 2000, "recall@1", "rrf_equal_3way", 0.304),
    ("biology", 100, "recall@20", "rsf_cv_equal", 0.376),
]
GUARD_TOL = 0.01

# Expected best-singleton median lines; only checked on a full run
# (ALL_SUBSETS × ALL_KS). best singleton = per-subset max of the three
# singletons, THEN cross-subset median.
BEST_SINGLETON_GUARD = {
    "recall@1": [0.340, 0.414, 0.366, 0.386, 0.376],
    "recall@20": [0.395, 0.468, 0.564, 0.588, 0.597],
}

# Per-domain plot lines: passthrough of per-subset matrix cells — no median.
PER_DOMAIN_DEFAULT = ["psychology", "robotics"]
PER_DOMAIN_METRICS = ["recall@1", "recall@20"]
# Spot-check vs published singletons (±GUARD_TOL).
PER_DOMAIN_GUARD = [
    ("psychology", 200, "recall@1", "cohere", 0.414),
    ("robotics", 2000, "recall@1", "zerank", 0.292),
]

_FILE_CACHE: dict[Path, dict] = {}


def _assert_not_caches(path: Path) -> None:
    assert "caches" not in path.parts, f"equal_weight must never read caches/: {path}"


def _load(path: Path) -> dict:
    """Load a run-summary JSON's `results` dict (memoized). Never touches caches/."""
    _assert_not_caches(path)
    if path not in _FILE_CACHE:
        with path.open() as fh:
            doc = json.load(fh)
        _FILE_CACHE[path] = doc
    return _FILE_CACHE[path]


def _metric_value(results: dict, cond: str, json_base: str, *, where: str) -> float:
    """Read results[cond]['avg_{json_base}_mean']; fail loud on any miss."""
    if cond not in results:
        raise KeyError(f"missing condition '{cond}' in {where}")
    entry = results[cond]
    key = f"avg_{json_base}_mean"
    if key not in entry:
        raise KeyError(f"missing metric '{key}' for condition '{cond}' in {where}")
    return float(entry[key])


def _has_all_equal_weight_keys(results: dict) -> bool:
    return all(k in results for k in equal_weight_keys())


def _subset_base(subset: str) -> Path:
    return RESULTS_DIR / f"bright_{subset}"


def resolve_rk20_source(subset: str, k: int) -> tuple[Path, str, dict]:
    """rk20 metrics: prefer runs/ if it has all 8 keys; else fall back to runs_rk100/.

    Detected by key presence, not a hard-coded subset list, so stub files and
    future re-collects resolve correctly.
    """
    runs_dir = _subset_base(subset) / "runs"
    for cand in sorted(runs_dir.glob(f"k{k}_from_k*.json")):
        doc = _load(cand)
        if doc.get("retrieved_k") != k:
            continue
        if _has_all_equal_weight_keys(doc["results"]):
            return cand, "runs", doc
    # rk100 sweep differs from the rk20 sweep only within the noise band.
    fb = _subset_base(subset) / "runs_rk100" / f"k{k}_from_k2000.json"
    if not fb.exists():
        raise FileNotFoundError(
            f"no rk20 source for {subset} k={k}: neither a complete runs/ file "
            f"nor {fb}"
        )
    doc = _load(fb)
    if doc.get("retrieved_k") != k:
        raise ValueError(f"{fb} has retrieved_k={doc.get('retrieved_k')}, expected {k}")
    if not _has_all_equal_weight_keys(doc["results"]):
        raise KeyError(f"{fb} is missing one or more equal-weight keys")
    return fb, "runs_rk100", doc


def resolve_rk100_source(subset: str, k: int) -> tuple[Path, str, dict]:
    """rk50/rk100 metrics: always the runs_rk100/ sweep for every subset."""
    path = _subset_base(subset) / "runs_rk100" / f"k{k}_from_k2000.json"
    if not path.exists():
        raise FileNotFoundError(f"missing rk100 source: {path}")
    doc = _load(path)
    if doc.get("retrieved_k") != k:
        raise ValueError(f"{path} has retrieved_k={doc.get('retrieved_k')}, expected {k}")
    if not _has_all_equal_weight_keys(doc["results"]):
        raise KeyError(f"{path} is missing one or more equal-weight keys")
    return path, "runs_rk100", doc


def build_rows(subsets: list[str], ks: list[int], metrics: list[str]) -> list[dict]:
    rows: list[dict] = []
    for subset in subsets:
        for k in ks:
            rk20_path, rk20_src, rk20_doc = resolve_rk20_source(subset, k)
            rk100_path = rk100_src = rk100_doc = None
            if any(m in RK100_METRICS for m in metrics):
                rk100_path, rk100_src, rk100_doc = resolve_rk100_source(subset, k)

            for metric in metrics:
                if metric in RK20_METRICS:
                    json_base, doc, src, path = (
                        RK20_METRICS[metric], rk20_doc, rk20_src, rk20_path,
                    )
                else:
                    assert rk100_doc is not None and rk100_path is not None
                    json_base, doc, src, path = (
                        RK100_METRICS[metric], rk100_doc, rk100_src, rk100_path,
                    )
                results = doc["results"]
                where = f"{path}"

                singletons = {
                    col: _metric_value(results, key, json_base, where=where)
                    for col, key in SINGLETON_KEYS.items()
                }
                for fusion in FUSIONS:
                    vals = {
                        col: _metric_value(
                            results, tmpl.format(f=fusion), json_base, where=where
                        )
                        for col, tmpl in FAMILY_KEYS.items()
                    }
                    cv, cz, vz, three = (
                        vals["cv_pair"], vals["cz_pair"], vals["vz_pair"], vals["threeway"],
                    )
                    best_pair = max(cv, cz, vz)
                    pair_lookup = {"cv": cv, "cz": cz, "vz": vz}
                    best_pair_name = ",".join(
                        name for name, v in pair_lookup.items()
                        if abs(v - best_pair) <= 1e-12
                    )
                    rows.append({
                        "subset": subset,
                        "k": k,
                        "metric": metric,
                        "fusion": fusion,
                        "cohere": singletons["cohere"],
                        "voyage": singletons["voyage"],
                        "zerank": singletons["zerank"],
                        # per-cell max; the report's line is per-subset max THEN
                        # cross-subset median — never max-of-medians.
                        "best_singleton": max(singletons.values()),
                        "cv_pair": cv,
                        "cz_pair": cz,
                        "vz_pair": vz,
                        "threeway": three,
                        "best_pair": best_pair,
                        "best_pair_name": best_pair_name,
                        "threeway_minus_best_pair": three - best_pair,
                        "threeway_minus_mean_pair": three - statistics.mean([cv, cz, vz]),
                        "source": src,
                        "source_file": str(path.relative_to(RESULTS_DIR)),
                    })
    return rows


def run_regression_guard() -> list[str]:
    notes: list[str] = []
    for subset, k, metric, cond, expected in REGRESSION_GUARD:
        json_base = ALL_METRICS[metric]
        if metric in RK20_METRICS:
            path, _, doc = resolve_rk20_source(subset, k)
        else:
            path, _, doc = resolve_rk100_source(subset, k)
        got = _metric_value(doc["results"], cond, json_base, where=str(path))
        if abs(got - expected) > GUARD_TOL:
            raise AssertionError(
                f"regression guard FAILED: {subset} k={k} {metric} {cond}: "
                f"got {got:.4f}, expected ~{expected:.3f} (tol {GUARD_TOL}) from {path}"
            )
        notes.append(
            f"  OK  {subset:13s} k={k:<4d} {metric:9s} {cond:16s} "
            f"got {got:.3f} ~ {expected:.3f}"
        )
    return notes


def run_best_singleton_guard(rows: list[dict], subsets: list[str],
                             ks: list[int]) -> list[str]:
    """Guard the `best singleton` line: definitional per-row max always, plus the
    full-run median lines against published values."""
    notes: list[str] = []
    for r in rows:
        mx = max(r["cohere"], r["voyage"], r["zerank"])
        if abs(r["best_singleton"] - mx) > 1e-12:
            raise AssertionError(
                f"best_singleton != max(singletons) at {r['subset']} k={r['k']} "
                f"{r['metric']}/{r['fusion']}: {r['best_singleton']} vs {mx}"
            )
    # Median lines only comparable to expectations on the full population.
    if subsets != ALL_SUBSETS or ks != ALL_KS:
        notes.append("  (best-singleton median guard skipped — partial run)")
        return notes
    for metric, expected in BEST_SINGLETON_GUARD.items():
        got = []
        for k in ALL_KS:
            # Singletons are fusion-independent → read the rrf rows only.
            vals = [r["best_singleton"] for r in rows
                    if r["metric"] == metric and r["fusion"] == "rrf"
                    and r["k"] == k]
            got.append(statistics.median(vals))
        for k, g, e in zip(ALL_KS, got, expected):
            if abs(g - e) > GUARD_TOL:
                raise AssertionError(
                    f"best-singleton guard FAILED: {metric} k={k}: "
                    f"got {g:.4f}, expected ~{e:.3f} (tol {GUARD_TOL})"
                )
        notes.append(f"  OK  best singleton {metric:9s} "
                     + " ".join(f"{g:.3f}" for g in got))
    return notes


def tag_delta(delta: float) -> str:
    """win / lose / ~tie per the ±0.01 noise band."""
    if delta > NOISE_BAND:
        return "win"
    if delta < -NOISE_BAND:
        return "lose"
    return "~tie"


def _sgn3(v: float) -> str:
    """Signed 3-decimal format, normalizing -0.000 → +0.000."""
    if abs(v) < 5e-4:
        v = 0.0
    return f"{v:+.3f}"


def fmt_delta(delta: float) -> str:
    return f"{_sgn3(delta)} {tag_delta(delta)}"


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def write_equal_weight_md(rows: list[dict], subsets: list[str], ks: list[int],
                          metrics: list[str]) -> Path:
    df = _df(rows)
    L: list[str] = []
    L.append("# Equal-weight 3-way vs. equal-weight pairs")
    L.append("")
    L.append("Holding weighting fixed at uniform, does the equal-weight 3-way "
             "(cohere+voyage+zerank, ⅓ each) beat the best equal-weight pair "
             "(cohere+voyage / cohere+zerank / voyage+zerank, 0.5/0.5)? RRF and "
             "RSF are kept strictly separate throughout.")
    L.append("")
    L.append("Generated by `analysis/equal_weight.py` — read-only over existing "
             "`runs*/` summary JSONs (zero reranker-API calls, caches/ never opened).")
    L.append("")

    L.append("## ⚠️ Caveats — read before citing")
    L.append("")
    if "robotics" in subsets:
        L.append("- **Robotics denominator mismatch.** On robotics the `vz` pair "
                 "and `threeway` columns average over fewer covered queries "
                 "(voyage+zerank & 3-way ≈ n=92) than the cohere-containing pairs "
                 "(`cv`/`cz` ≈ n=96/101). A robotics 3way-vs-cz delta therefore "
                 "mixes a denominator difference into the comparison — not fixed "
                 "here (read JSON values as-is).")
    L.append(f"- **Noise floor.** 1 query ≈ 0.010 on R@1. Deltas with "
             f"`|threeway − best_pair| ≤ {NOISE_BAND}` are tagged `~tie` and no "
             "winner is declared inside that band. **RSF-equal cells** are the "
             "most tie-prone in the menu (uniform weights + min-max normalization "
             "→ exact fused-score ties → ~1-query PYTHONHASHSEED wobble), so "
             "sub-0.02 RSF deltas may be tie-break artifacts; RRF-equal and "
             "singletons are exactly reproducible.")
    L.append("- **RRF and RSF are never pooled** into a single number anywhere "
             "in this report.")
    L.append("")

    L.append("## (a) Per-metric headline tables")
    L.append("")
    L.append("Columns: `cv` (cohere+voyage) · `cz` (cohere+zerank) · `vz` "
             "(voyage+zerank) · `3way` · `best_pair` · `3way−best_pair` "
             "(tagged win/lose/~tie). One table per fusion method. Rounded to 3 "
             "decimals; deltas computed at full precision.")
    L.append("")
    ordered_metrics = [m for m in METRIC_ORDER if m in metrics]
    for metric in ordered_metrics:
        L.append(f"### {metric}")
        L.append("")
        for fusion in FUSIONS:
            sub = df[(df["metric"] == metric) & (df["fusion"] == fusion)]
            if sub.empty:
                continue
            rsf_note = " — RSF: sub-0.02 deltas may be tie-break artifacts" if fusion == "rsf" else ""
            L.append(f"**{fusion.upper()}**{rsf_note}")
            L.append("")
            L.append("| subset | k | cv | cz | vz | 3way | best_pair | 3way−best_pair |")
            L.append("|---|---|---|---|---|---|---|---|")
            for subset in subsets:
                for k in ks:
                    r = sub[(sub["subset"] == subset) & (sub["k"] == k)]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    L.append(
                        f"| {subset} | {k} | {r['cv_pair']:.3f} | "
                        f"{r['cz_pair']:.3f} | {r['vz_pair']:.3f} | "
                        f"{r['threeway']:.3f} | {r['best_pair']:.3f} "
                        f"({r['best_pair_name']}) | {fmt_delta(r['threeway_minus_best_pair'])} |"
                    )
            L.append("")

    L.append("## (b) Thesis summary — `threeway_minus_best_pair` per (subset, metric, fusion)")
    L.append("")
    L.append("Per-k vector of `3way − best_pair` and the median across k. "
             "win = `> +0.01`, lose = `< −0.01`, ~tie = `|·| ≤ 0.01` (noise). "
             "This is the table the section's claims cite.")
    L.append("")
    for fusion in FUSIONS:
        L.append(f"### {fusion.upper()}")
        L.append("")
        kcols = " | ".join(f"k={k}" for k in ks)
        L.append(f"| subset | metric | {kcols} | median-k | verdict |")
        L.append("|---|---|" + "---|" * (len(ks) + 2))
        for subset in subsets:
            for metric in ordered_metrics:
                sub = df[(df["subset"] == subset) & (df["metric"] == metric)
                         & (df["fusion"] == fusion)]
                if sub.empty:
                    continue
                by_k = {int(row["k"]): row["threeway_minus_best_pair"]
                        for _, row in sub.iterrows()}
                vec = [by_k.get(k) for k in ks]
                cells = " | ".join(
                    _sgn3(v) if v is not None else "—" for v in vec
                )
                present = [v for v in vec if v is not None]
                med = statistics.median(present) if present else float("nan")
                L.append(
                    f"| {subset} | {metric} | {cells} | {_sgn3(med)} | "
                    f"{tag_delta(med)} |"
                )
        L.append("")

    L.append("## (c) Cross-domain aggregate — median of `threeway_minus_best_pair` across subsets")
    L.append("")
    L.append("Median across all subsets of `3way − best_pair` per (metric, fusion, k).")
    L.append("")
    for fusion in FUSIONS:
        L.append(f"### {fusion.upper()}")
        L.append("")
        kcols = " | ".join(f"k={k}" for k in ks)
        L.append(f"| metric | {kcols} |")
        L.append("|---|" + "---|" * len(ks))
        for metric in ordered_metrics:
            cells = []
            for k in ks:
                vals = df[(df["metric"] == metric) & (df["fusion"] == fusion)
                          & (df["k"] == k)][
                    "threeway_minus_best_pair"].tolist()
                cells.append(_sgn3(statistics.median(vals)) if vals else "—")
            L.append(f"| {metric} | " + " | ".join(cells) + " |")
        L.append("")

    L.append("## (d) Absolute lines for plotting — singletons, pairs, and the 3-way")
    L.append("")
    L.append("Each condition's **absolute** metric value, as the **cross-subset "
             "median** of that condition (the lines you plot). Includes the three "
             "individual cross-encoders (`cohere` / `voyage` / `zerank`), the three "
             "equal-weight pairs (`cv` / `cz` / `vz`), the equal-weight `3way`, and "
             "`best_pair` (the per-cell winning pair's value — its identity can "
             "change per cell; see Section a). The **singleton lines are fusion-"
             "independent** (a single reranker doesn't fuse), so the `cohere` / "
             "`voyage` / `zerank` rows are identical under RRF and RSF. NOTE: each "
             "condition's median is computed independently, so "
             "`median(3way) − median(vz)` is **not** the per-cell "
             "`threeway_minus_best_pair` median in Section (b)/(c) — medians don't commute "
             "with subtraction. Use these rows for absolute-value figures; use "
             "Section (b)/(c) for the paired delta. `best singleton` is the per-subset max "
             "of the three singletons, then the cross-subset median (computed at "
             "full precision — NOT the max of the three already-medianed singleton "
             "lines).")
    L.append("")
    if "robotics" in subsets:
        L.append("> **Robotics denominator note.** Robotics' voyage/zerank "
                 "singletons average over ~96 covered queries vs cohere's 101 "
                 "(read as-is from `runs*/`, not re-intersected).")
        L.append("")
    line_cols = [("cohere", "cohere"), ("voyage", "voyage"), ("zerank", "zerank"),
                 ("best singleton", "best_singleton"),
                 ("cv", "cv_pair"), ("cz", "cz_pair"), ("vz", "vz_pair"),
                 ("3way", "threeway"), ("best_pair", "best_pair")]
    for metric in ordered_metrics:
        L.append(f"### {metric}")
        L.append("")
        for fusion in FUSIONS:
            sub = df[(df["metric"] == metric) & (df["fusion"] == fusion)]
            if sub.empty:
                continue
            kcols = " | ".join(f"k={k}" for k in ks)
            L.append(f"**{fusion.upper()}** — cross-subset median per condition")
            L.append("")
            L.append(f"| line | {kcols} |")
            L.append("|---|" + "---|" * len(ks))
            for label, col in line_cols:
                cells = []
                for k in ks:
                    vals = sub[(sub["k"] == k)][col].tolist()
                    cells.append(f"{statistics.median(vals):.3f}" if vals else "—")
                L.append(f"| {label} | " + " | ".join(cells) + " |")
            L.append("")
    L.append("")

    L.append("## One-line claims (operating point k=200)")
    L.append("")
    op_k = 200 if 200 in ks else ks[len(ks) // 2]
    for metric in [m for m in ["recall@1", "recall@20"] if m in metrics]:
        for fusion in FUSIONS:
            vals = df[(df["metric"] == metric) & (df["fusion"] == fusion)
                      & (df["k"] == op_k)][
                "threeway_minus_best_pair"].tolist()
            if not vals:
                continue
            med = statistics.median(vals)
            verb = {"win": "beats", "lose": "loses to", "~tie": "ties"}[tag_delta(med)]
            L.append(
                f"- **{metric} / {fusion.upper()} @ k={op_k} (all subsets):** "
                f"equal-weight 3-way {verb} the best equal-weight pair by a median "
                f"of {_sgn3(med)} across subsets."
            )
    L.append("")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "EQUAL_WEIGHT.md"
    out.write_text("\n".join(L))
    return out


def write_wide_table_md(rows: list[dict]) -> Path:
    df = _df(rows)
    cols = ["subset", "k", "metric", "fusion", "cohere", "voyage", "zerank",
            "best_singleton", "cv_pair", "cz_pair", "vz_pair", "threeway",
            "best_pair", "best_pair_name", "threeway_minus_best_pair",
            "threeway_minus_mean_pair", "source"]
    df = df[cols].sort_values(["metric", "fusion", "subset", "k"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "equal_weight_table.md"
    lines = ["# Equal-weight matrix (wide dump for spot-checking)", ""]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    out.write_text("\n".join(lines))
    return out


def write_per_domain_lines(rows: list[dict], want_subsets: list[str],
                           metrics: list[str]) -> tuple[Path, Path] | None:
    """Emit each requested domain's own plot lines (passthrough, no median).

    Pairs/3-way are read at RSF; singletons are fusion-independent.
    `fusion_over_best_singleton` = best equal pair (RSF) − best singleton — the
    uniform-weight floor, smaller than the best-fusion lift.
    """
    df = _df(rows)
    out_subsets = [s for s in want_subsets
                   if s in set(df["subset"]) ]
    use_metrics = [m for m in PER_DOMAIN_METRICS if m in metrics]
    if not out_subsets or not use_metrics:
        print("per-domain-lines: nothing to emit (requested subsets/metrics "
              f"absent from this run: want={want_subsets}, have R@1/R@20={use_metrics})")
        return None

    # Spot-check published singletons (only the cells present).
    for subset, k, metric, col, expected in PER_DOMAIN_GUARD:
        sel = df[(df["subset"] == subset) & (df["k"] == k) & (df["metric"] == metric)
                 & (df["fusion"] == "rsf")]
        if sel.empty:
            continue
        got = float(sel.iloc[0][col])
        if abs(got - expected) > GUARD_TOL:
            raise AssertionError(
                f"per-domain guard FAILED: {subset} k={k} {metric} {col}: "
                f"got {got:.4f}, expected ~{expected:.3f} (tol {GUARD_TOL})"
            )

    records: list[dict] = []
    for subset in out_subsets:
        for metric in use_metrics:
            for k in sorted({int(x) for x in df["k"].unique()}):
                sel = df[(df["subset"] == subset) & (df["k"] == k)
                         & (df["metric"] == metric) & (df["fusion"] == "rsf")]
                if sel.empty:
                    raise ValueError(
                        f"per-domain-lines: missing RSF row for {subset} k={k} {metric}"
                    )
                r = sel.iloc[0]
                records.append({
                    "subset": subset,
                    "metric": metric,
                    "k": k,
                    "cohere": float(r["cohere"]),
                    "voyage": float(r["voyage"]),
                    "zerank": float(r["zerank"]),
                    "best_singleton": float(r["best_singleton"]),
                    "best_pair_rsf": float(r["best_pair"]),
                    "best_pair_name_rsf": str(r["best_pair_name"]),
                    "threeway_rsf": float(r["threeway"]),
                    "fusion_over_best_singleton": float(r["best_pair"]) - float(r["best_singleton"]),
                })

    # Most/least sanity: psychology's R@1 effect must clearly exceed robotics's.
    # Uses the across-k PEAK — the equal-RSF effect is depth-concentrated.
    def _r1_peak(sub: str):
        e = [rec["fusion_over_best_singleton"] for rec in records
             if rec["subset"] == sub and rec["metric"] == "recall@1"]
        return max(e) if e else None
    psy_pk, rob_pk = _r1_peak("psychology"), _r1_peak("robotics")
    if psy_pk is not None and rob_pk is not None:
        if psy_pk < rob_pk + 0.01:
            raise AssertionError(
                "per-domain most/least sanity FAILED: psychology R@1 peak effect "
                f"{psy_pk:+.3f} not clearly above robotics {rob_pk:+.3f} — a wrong "
                "column is likely being read."
            )
        print(f"per-domain-lines: R@1 effect peak psychology {psy_pk:+.3f} > "
              f"robotics {rob_pk:+.3f} (most>least OK)")

    out_dir = OUT_DIR / "per_domain_lines"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = pd.DataFrame.from_records(records)
    csv_path = out_dir / "per_domain_lines.csv"
    pdf.to_csv(csv_path, index=False)

    L: list[str] = ["# Per-domain plot lines (3-row fusion figure)", ""]
    L.append("Each domain's own five lines (`cohere`/`voyage`/`zerank` singletons, "
             "`best_pair_rsf`, `threeway_rsf`) plus `best_singleton` (per-cell "
             "max of the three singletons) and the signed effect "
             "`fusion_over_best_singleton` = `best_pair_rsf − best_singleton`. "
             "Pairs/3-way at **RSF** (the figure's fusion); singletons are "
             "fusion-independent. Passthrough of per-subject matrix cells — no "
             "median. RSF effect under ±0.01 tagged `~tie`.")
    L.append("")
    L.append("> **Effect definition (figure caption).** Psychology/Robotics were "
             "chosen as most/least by the documented per-cell **best-fusion** R@1 "
             "lift (+0.078 / +0.010–0.013, Section \"Cross-domain\"). The column here is "
             "**best *equal* pair − best singleton** (uniform weight), which is "
             "**smaller** than that headline because it excludes weight tuning — "
             "expected, not a discrepancy. The panels show the uniform-weight floor.")
    L.append("")
    L.append("> **Most/least, honestly.** Psychology's equal-RSF R@1 effect is "
             "**depth-concentrated** — peak +0.038 at k=500, but ~0.000 at k=200, "
             "where cohere *alone* (0.414) is the domain's published R@1 winner and "
             "the equal `cv` pair only ties it. Psychology's +0.078 headline came "
             "from a weight-tuned RRF condition outside the equal-weight menu "
             "(a historical tilted blend), not equal RSF. Robotics is flat "
             "~+0.010. So psychology > robotics holds by peak/median across k, not "
             "at every individual k.")
    L.append("")
    if "robotics" in out_subsets:
        L.append("> **Robotics coverage note.** Coverage denominators differ — "
                 "cohere ~101 queries, voyage/zerank ~96, vz/3-way ~92 — read "
                 "as-is, not re-intersected.")
        L.append("")
    for subset in out_subsets:
        L.append(f"## {subset}")
        L.append("")
        for metric in use_metrics:
            L.append(f"### {metric}")
            L.append("")
            L.append("| k | cohere | voyage | zerank | best_singleton | "
                     "best_pair_rsf | threeway_rsf | fusion_over_best_singleton |")
            L.append("|---|---|---|---|---|---|---|---|")
            for rec in [x for x in records if x["subset"] == subset and x["metric"] == metric]:
                eff = rec["fusion_over_best_singleton"]
                eff_str = f"{_sgn3(eff)}" + (" ~tie" if abs(eff) <= NOISE_BAND else "")
                L.append(
                    f"| {rec['k']} | {rec['cohere']:.3f} | {rec['voyage']:.3f} | "
                    f"{rec['zerank']:.3f} | {rec['best_singleton']:.3f} | "
                    f"{rec['best_pair_rsf']:.3f} ({rec['best_pair_name_rsf']}) | "
                    f"{rec['threeway_rsf']:.3f} | {eff_str} |"
                )
            L.append("")
    md_path = out_dir / "per_domain_lines.md"
    md_path.write_text("\n".join(L))
    return csv_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subsets", nargs="+", choices=ALL_SUBSETS, default=None,
                        help="Subsets to include (default: all five).")
    parser.add_argument("--k", nargs="+", type=int, choices=ALL_KS, default=None,
                        help="retrieved_k operating point(s) (default: all).")
    parser.add_argument("--smoke", action="store_true",
                        help="biology only, k=200, R@1+R@20.")
    parser.add_argument("--per-domain-lines", nargs="*", choices=ALL_SUBSETS,
                        default=None, metavar="SUBSET",
                        help="Emit per-domain plot lines (R@1/R@20, five lines each) "
                             "for these subsets (default: psychology robotics). Always "
                             "emitted on a full run for those two.")
    args = parser.parse_args()

    if args.smoke:
        subsets, ks, metrics = ["biology"], [200], ["recall@1", "recall@20"]
    else:
        subsets = args.subsets or ALL_SUBSETS
        ks = args.k or ALL_KS
        metrics = list(ALL_METRICS.keys())

    assert "caches" not in OUT_DIR.parts, "equal_weight output must not live under caches/"

    print(f"subsets={subsets}  ks={ks}  metrics={metrics}")
    print("running regression guard (reads must match CLAUDE.md published winners)...")
    for note in run_regression_guard():
        print(note)

    rows = build_rows(subsets, ks, metrics)
    print(f"built {len(rows)} matrix rows (subset×k×metric×fusion)")

    print("running best-singleton guard (per-row max + full-run median lines)...")
    for note in run_best_singleton_guard(rows, subsets, ks):
        print(note)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = _df(rows)

    matrix_json = OUT_DIR / "equal_weight_matrix.json"
    with matrix_json.open("w") as fh:
        json.dump(rows, fh, indent=2)
    matrix_parquet = OUT_DIR / "equal_weight_matrix.parquet"
    df.to_parquet(matrix_parquet, index=False)

    md = write_equal_weight_md(rows, subsets, ks, metrics)
    wide = write_wide_table_md(rows)

    written = [matrix_json, matrix_parquet, md, wide]

    per_domain_want = args.per_domain_lines if args.per_domain_lines else PER_DOMAIN_DEFAULT
    pd_paths = write_per_domain_lines(rows, per_domain_want, metrics)
    if pd_paths is not None:
        written.extend(pd_paths)

    print("wrote:")
    for p in written:
        print(f"  {p.relative_to(RESULTS_DIR.parent)}")


if __name__ == "__main__":
    main()
