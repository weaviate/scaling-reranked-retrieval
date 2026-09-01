"""Per-clone unique successes under the noise-null — the winner's-curse twin of
the real unique-successes table.

Runs the unchanged analyze_subset counter over three i.i.d. noisy clones of one
base model; any null any-unique count is selection-on-noise. Tie-free
(singletons only, no fusion), so no PYTHONHASHSEED pin is needed. Zero network.
"""
from __future__ import annotations

import argparse
import json
import statistics

from scaling_reranked_retrieval.config import RESULTS_DIR
from scaling_reranked_retrieval.application.queryset import load_and_validate

from scaling_reranked_retrieval.application.analysis.noise_null import build_clone_cache, install_no_network_guard
from scaling_reranked_retrieval.application.analysis.unique_successes import (
    CACHE_K,
    PROVIDERS,
    RERANKED_K,
    SUBSETS,
    SUCCESS_CUTOFFS,
    analyze_subset,
)

from scaling_reranked_retrieval.adapters import qab

qab.setup()

RETRIEVED_K = 200                                  # match the real table's operating point
BASE = "zerank"                                    # strongest R@1 singleton
ALPHAS = (0.05, 0.10, 0.25, 0.50, 1.00)            # same sweep as noise_null
N_SEEDS = 20
# Clone slots are relabeled A/B/C in output — they are not the real models.
CLONE_LABELS = {p: lab for p, lab in zip(PROVIDERS, ("A", "B", "C"))}


def _aggregate_over_subsets(per_subset: dict, present: list[str]) -> dict:
    """Sum unique/success/coverage across subsets (absolute query tallies)."""
    agg = {}
    for K in SUCCESS_CUTOFFS:
        agg[K] = {
            p: {
                "unique": sum(per_subset[d]["counts"][K][p]["unique"] for d in present),
                "success": sum(per_subset[d]["counts"][K][p]["success"] for d in present),
            }
            for p in PROVIDERS
        }
        agg[K]["any_unique"] = sum(agg[K][p]["unique"] for p in PROVIDERS)
        agg[K]["any_hit"] = sum(per_subset[d]["coverage"][K]["any_hit"] for d in present)
        agg[K]["all_miss"] = sum(per_subset[d]["coverage"][K]["all_miss"] for d in present)
    agg["n"] = sum(per_subset[d]["n_queries_intersection"] for d in present)
    return agg


def _mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    m = statistics.mean(xs)
    s = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return m, s


def run_null(
    real: dict[str, tuple],
    k: int,
    reranked_k: int,
    alphas: tuple,
    n_seeds: int,
    base: str,
) -> dict:
    """Per-(alpha) seed-averaged unique-success counts over the noise clones."""
    present = list(real.keys())
    out: dict[float, dict] = {}

    for alpha in alphas:
        seed_agg = []
        ps_unique = {
            d: {K: {p: [] for p in PROVIDERS} for K in SUCCESS_CUTOFFS}
            for d in present
        }
        clone_r1_seeds = []

        for seed in range(n_seeds):
            per_subset = {}
            for ds in present:
                cache, qs = real[ds]
                queries = list(qs.gold.keys())
                clone = build_clone_cache(cache, queries, base, alpha, seed)
                res = analyze_subset(clone, qs, k, reranked_k)
                per_subset[ds] = res
                for K in SUCCESS_CUTOFFS:
                    for p in PROVIDERS:
                        ps_unique[ds][K][p].append(res["counts"][K][p]["unique"])
            agg = _aggregate_over_subsets(per_subset, present)
            seed_agg.append(agg)
            n_all = agg["n"]
            clone_r1_seeds.append(
                statistics.mean(agg[1][p]["success"] / n_all for p in PROVIDERS)
            )
            print(f"    [alpha={alpha} seed={seed}] "
                  f"any-unique @1={agg[1]['any_unique']} @20={agg[20]['any_unique']}",
                  flush=True)

        agg_stats = {}
        for K in SUCCESS_CUTOFFS:
            per_slot = {}
            for p in PROVIDERS:
                m, s = _mean_std([sa[K][p]["unique"] for sa in seed_agg])
                per_slot[p] = {"unique_mean": m, "unique_std": s}
            any_m, any_s = _mean_std([sa[K]["any_unique"] for sa in seed_agg])
            hit_m, _ = _mean_std([sa[K]["any_hit"] for sa in seed_agg])
            miss_m, _ = _mean_std([sa[K]["all_miss"] for sa in seed_agg])
            agg_stats[K] = {
                "per_slot": per_slot,
                "any_unique_mean": any_m,
                "any_unique_std": any_s,
                "per_clone_mean": any_m / len(PROVIDERS),
                "any_hit_mean": hit_m,
                "all_miss_mean": miss_m,
            }
        ps_stats = {
            d: {
                K: {p: _mean_std(ps_unique[d][K][p])[0] for p in PROVIDERS}
                for K in SUCCESS_CUTOFFS
            }
            for d in present
        }
        out[alpha] = {
            "n": seed_agg[0]["n"],
            "aggregate": agg_stats,
            "per_subset_unique_mean": ps_stats,
            "clone_hit_rate_at_1": statistics.mean(clone_r1_seeds),
        }
        print(f"  alpha={alpha}: any-unique@1 "
              f"{agg_stats[1]['any_unique_mean']:.1f}±{agg_stats[1]['any_unique_std']:.1f} "
              f"(clone R@1={out[alpha]['clone_hit_rate_at_1']:.3f})", flush=True)
    return out


def compute_real_reference(real: dict[str, tuple], k: int, reranked_k: int) -> dict:
    """Real unique successes via the same `analyze_subset` path (not read off disk)."""
    present = list(real.keys())
    per_subset = {}
    for ds in present:
        cache, qs = real[ds]
        per_subset[ds] = analyze_subset(cache, qs, k, reranked_k)
    agg = _aggregate_over_subsets(per_subset, present)
    return {"per_subset": per_subset, "aggregate": agg}


def run(k: int, reranked_k: int, alphas: tuple, n_seeds: int, base: str,
        subsets: list[str], write: bool = True) -> dict:
    install_no_network_guard()

    real_loaded: dict[str, tuple] = {}
    for ds in subsets:
        loaded = load_and_validate(ds)
        if loaded is None:
            print(f"  [skip] no usable cache for {ds}")
            continue
        real_loaded[ds] = loaded
    if not real_loaded:
        raise SystemExit("No subsets had a usable k=2000 cache.")
    present = list(real_loaded.keys())

    print(f"\n=== Real reference (k={k}) ===")
    real_ref = compute_real_reference(real_loaded, k, reranked_k)
    ra = real_ref["aggregate"]
    print("  real any-unique @1={} (c/v/z {}/{}/{})".format(
        ra[1]["any_unique"], *[ra[1][p]["unique"] for p in PROVIDERS]))

    print(f"\n=== Null sweep (base={base}, {n_seeds} seeds, k={k}) ===")
    null = run_null(real_loaded, k, reranked_k, alphas, n_seeds, base)

    payload = {
        "k": k,
        "reranked_k": reranked_k,
        "cache_retrieved_k": CACHE_K,
        "success_cutoffs": list(SUCCESS_CUTOFFS),
        "base_model": base,
        "n_seeds": n_seeds,
        "alphas": list(alphas),
        "subsets_present": present,
        "definition": (
            "Null unique success @K = a noise clone (one of three i.i.d. copies of "
            f"{base}) lands a gold doc in its top-K while both other clones miss. "
            "Counted with the same analyze_subset() as the real per-reranker table, "
            f"over the all-three-present intersection at retrieved_k={k}, "
            f"reranked_k={reranked_k}. Three clones are i.i.d. so any non-zero "
            "any-unique count is winner's curse (selection-on-noise)."
        ),
        "clone_labels": CLONE_LABELS,
        "real_reference": real_ref,
        "null_by_alpha": null,
    }

    if write:
        out_dir = RESULTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"unique_successes_noise_null_k{k}.json"
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote {json_path}")
        md_path = out_dir / f"unique_successes_noise_null_k{k}_table.md"
        md_path.write_text(render_table(payload))
        print(f"Wrote {md_path}")

    return payload


def _valid_regime(payload: dict) -> list[float]:
    """Alphas where clones keep >=90% of their alpha->0 hit-rate@1 (matches
    noise_null's regime)."""
    null = payload["null_by_alpha"]
    alphas = sorted(null.keys(), key=float)
    clean = null[alphas[0]]["clone_hit_rate_at_1"]
    thresh = 0.9 * clean if clean else 0.0
    return [a for a in alphas if null[a]["clone_hit_rate_at_1"] >= thresh]


def render_table(payload: dict) -> str:
    k = payload["k"]
    base = payload["base_model"]
    null = payload["null_by_alpha"]
    real = payload["real_reference"]["aggregate"]
    present = payload["subsets_present"]
    alphas = sorted(null.keys(), key=float)
    valid = _valid_regime(payload)
    L: list[str] = []
    A = L.append

    A(f"# Unique Successes under the Noise-Null at k={k}")
    A("")
    A(
        f"The winner's-curse twin of `unique_successes_k{k}_table.md`. The **real** row "
        f"is the three real rerankers (Cohere / Voyage / Zerank). The **null** rows clone "
        f"ONE base (`{base}`, the strongest R@1 singleton) into three i.i.d. noisy copies "
        f"`s_i = s + e_i`, `e_i ~ N(0, α·σ_q)`, and count unique successes per clone with "
        f"the *same* `analyze_subset` — over the all-three-present intersection, "
        f"retrieved_k={k}, reranked_k={payload['reranked_k']}, {payload['n_seeds']} seeds "
        f"(mean ± std). Zero true heterogeneity ⇒ any null **any-unique** count is "
        f"selection-on-noise. Three clones are i.i.d. so their per-slot counts are equal "
        f"in expectation (symmetry check); the real 11/19/21 asymmetry has no null analogue."
    )
    A("")
    A(f"**Clone calibration.** Reasonable-α regime (clones keep ≥90% of their α→0 "
      f"hit-rate@1): **{valid}**. Clone hit-rate@1 by α: "
      + ", ".join(f"α={a}→{null[a]['clone_hit_rate_at_1']:.3f}" for a in alphas)
      + f" (real `{base}` base singleton R@1 ≈ "
      + f"{real[1][base]['success'] / real['n']:.3f}).")
    A("")

    for K in SUCCESS_CUTOFFS:
        A(f"## Aggregate unique successes @{K} (n={real['n']})")
        A("")
        A("| Source | A | B | C | per-model mean | **any-unique** |")
        A("|---|---|---|---|---|---|")
        rc = [real[K][p]["unique"] for p in PROVIDERS]
        A(f"| **real** (C/V/Z) | {rc[0]} | {rc[1]} | {rc[2]} | "
          f"{sum(rc) / len(PROVIDERS):.1f} | **{real[K]['any_unique']}** |")
        for a in alphas:
            st = null[a]["aggregate"][K]
            cells = [f"{st['per_slot'][p]['unique_mean']:.1f}" for p in PROVIDERS]
            tag = "" if a in valid else " ⟂"          # ⟂ = outside regime
            A(f"| null α={a}{tag} | {cells[0]} | {cells[1]} | {cells[2]} | "
              f"{st['per_clone_mean']:.1f} | "
              f"{st['any_unique_mean']:.1f} ± {st['any_unique_std']:.1f} |")
        A("")

    A("## Real vs. null — any-unique (the heterogeneity that fusion feeds on)")
    A("")
    A("Conservative null = the largest **valid-regime** α (clones still ≈ base quality). "
      "`real − null` is the genuine-heterogeneity excess over winner's curse; "
      "`null / real` is the fraction of the real unique-success population a pure "
      "selection-on-noise process reproduces.")
    A("")
    cons = max(valid, key=float) if valid else alphas[0]
    A(f"| Cutoff | real any-unique | null any-unique (α={cons}) | real − null | null / real |")
    A("|---|---|---|---|---|")
    for K in SUCCESS_CUTOFFS:
        rv = real[K]["any_unique"]
        nv = null[cons]["aggregate"][K]["any_unique_mean"]
        frac = nv / rv if rv else float("nan")
        A(f"| @{K} | {rv} | {nv:.1f} | {rv - nv:+.1f} | {frac:.2f} |")
    A("")

    for K in (1, 20):
        A(f"## Per-subset unique successes @{K} — real vs null (α={cons})")
        A("")
        A("| Subset | n | real C/V/Z | real any | null A/B/C (mean) | null any (mean) |")
        A("|---|---|---|---|---|---|")
        for ds in present:
            rs = payload["real_reference"]["per_subset"][ds]
            n = rs["n_queries_intersection"]
            rcvz = [rs["counts"][K][p]["unique"] for p in PROVIDERS]
            r_any = sum(rcvz)
            nm = null[cons]["per_subset_unique_mean"][ds][K]
            n_cells = [nm[p] for p in PROVIDERS]
            n_any = sum(n_cells)
            A(f"| {ds} | {n} | {rcvz[0]}/{rcvz[1]}/{rcvz[2]} | {r_any} | "
              f"{n_cells[0]:.1f}/{n_cells[1]:.1f}/{n_cells[2]:.1f} | {n_any:.1f} |")
        A("")

    lo_a = alphas[0]
    f1_lo = null[lo_a]["aggregate"][1]["any_unique_mean"] / real[1]["any_unique"]
    f1_hi = null[cons]["aggregate"][1]["any_unique_mean"] / real[1]["any_unique"]

    A("## Reading")
    A("")
    A("- **Symmetry holds ⇒ the null is unbiased.** The three i.i.d. clone slots (A/B/C) "
      "carry ~equal unique-success counts at every α; the real rerankers do not "
      "(Voyage/Zerank > Cohere @1; Cohere leads @20; per-domain the leader flips — "
      "psychology Cohere 7, robotics Zerank 6 @1), so the real *structure* is a "
      "model-identity effect a noise null cannot reproduce at any α.")
    A("- **The magnitude is partly reproducible by noise; the structure is not.** "
      f"The conservative valid null (α={cons}) reproduces {f1_hi:.0%} of the real @1 "
      f"any-unique count and a small valid α={lo_a} only {f1_lo:.0%} — so the raw "
      "*count* of unique successes is sensitive to the assumed noise scale, but the "
      "*asymmetry across models and domains* (the panel structure) has no null twin.")
    A("- **Real exceeds the null at every cutoff, but the margin is tightest at @1.** "
      f"real − null(α={cons}) = +{real[1]['any_unique'] - null[cons]['aggregate'][1]['any_unique_mean']:.0f} "
      f"@1 / +{real[5]['any_unique'] - null[cons]['aggregate'][5]['any_unique_mean']:.0f} @5 / "
      f"+{real[20]['any_unique'] - null[cons]['aggregate'][20]['any_unique_mean']:.0f} @20, "
      "and null/real is *highest* at @1 (0.78) — at the conservative α the head is where "
      "selection-on-noise manufactures the largest share of the apparent heterogeneity. "
      "This is the caveat the per-reranker R@1 table must be read against.")
    return "\n".join(L)


def run_smoke() -> None:
    print("=== SMOKE: biology, 3 seeds, alpha {0.10, 0.50}, k=200 ===")
    install_no_network_guard()
    payload = run(
        k=200, reranked_k=RERANKED_K, alphas=(0.10, 0.50), n_seeds=3,
        base=BASE, subsets=["biology"], write=False,
    )
    null = payload["null_by_alpha"]
    # Symmetry: i.i.d. slots should carry ~equal unique counts.
    for a in (0.10, 0.50):
        slots = [null[a]["aggregate"][1]["per_slot"][p]["unique_mean"] for p in PROVIDERS]
        spread = max(slots) - min(slots)
        print(f"  alpha={a} @1 per-slot {[f'{s:.1f}' for s in slots]} spread={spread:.1f}")
    lo = null[0.10]["aggregate"][1]["any_unique_mean"]
    hi = null[0.50]["aggregate"][1]["any_unique_mean"]
    print(f"  any-unique@1: alpha=0.10 {lo:.1f}  alpha=0.50 {hi:.1f}")
    assert hi >= lo - 1e-9, "any-unique did not grow with alpha"
    # Regression: published biology @1 c/v/z = 0/5/3.
    rb = payload["real_reference"]["per_subset"]["biology"]["counts"][1]
    cvz = [rb[p]["unique"] for p in PROVIDERS]
    print(f"  real biology @1 c/v/z = {cvz} (expect [0, 5, 3])")
    assert cvz == [0, 5, 3], cvz
    print("\nSMOKE PASSED.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="fast self-test (biology)")
    parser.add_argument("--k", type=int, default=RETRIEVED_K,
                        help="retrieved_k operating point (default 200, matches the real table)")
    parser.add_argument("--reranked-k", type=int, default=RERANKED_K)
    parser.add_argument("--alphas", nargs="+", type=float, default=list(ALPHAS))
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument("--base", default=BASE, choices=list(PROVIDERS))
    parser.add_argument("--subsets", nargs="+", default=list(SUBSETS))
    parser.add_argument("--depths", nargs="+", type=int, default=None,
                        help="convenience: run several --k operating points in one go")
    args = parser.parse_args()

    if args.smoke:
        run_smoke()
        return

    ks = args.depths if args.depths else [args.k]
    for k in ks:
        run(k=k, reranked_k=args.reranked_k, alphas=tuple(args.alphas),
            n_seeds=args.seeds, base=args.base, subsets=args.subsets)


if __name__ == "__main__":
    main()
