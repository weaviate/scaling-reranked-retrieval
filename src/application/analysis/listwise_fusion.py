"""RRF fusion analysis over cached listwise-LLM rankings (zero LLM calls).

The CE experiment's fusion question, ported to the listwise tier: do the
listwise rerankers (gpt-5.4-mini, gpt-5.6-luna, gpt-5.6-terra — the
effort-`none` trio) make sufficiently different errors — from each other and
from the pool-source cross-encoder (Zerank-2) — that fusing their rankings
beats the best single ranker on the head metrics?

Method: RRF only (k=60). Listwise rerankers emit rankings, not scores, so
rank-based fusion is the principled choice; a score-based RSF analog over
rank-derived scores degenerates to weighted Borda and is deliberately not
reported (see the design discussion in CLAUDE.md / ARCHITECTURE.md).

Inputs (all on disk; no network):
  - results/listwise/pools/<domain>__<pool-slug>__first<K>__top<P>.json
      (fixed pool, its baseline order = the pool-source CE ranking, gold sets)
  - results/listwise/cache/<domain>__<model>__<effort>__<pool-slug>__...jsonl
      (per-(query, trial) listwise rankings, loaded via src.listwise)

Rankers fused: the pool-source baseline (constant across trials) + one ranker
per --models entry. Trials policy: TRIAL-ALIGNED — trial i of every stochastic
model is fused with trial i of the others (zerank constant), giving one fused
result per trial; report mean ± std across trials, matching the listwise
experiment's own variance methodology.

Menu (per available ranker set): singletons; every subset of size >= 2 at
equal weight, member-named (with base + 3 models: 6 pairs, 4 triples, the
4-way — equal-weight only — the experiment-wide menu policy,
see src.conditions). Metrics: R@1, R@5,
nDCG@10 (mean across queries per subset; median across subsets aggregate).
R@POOL_K is the pool
ceiling — invariant under reordering — and is reported once per subset.

Outputs: results/listwise/fusion/<pool-slug>__first<K>__top<P>/
    fusion.json     full per-subset × per-condition × per-trial results
    FUSION.md       cross-subset report

Usage:
    uv run python scripts/listwise_fusion.py                       # both models
    uv run python scripts/listwise_fusion.py --models gpt-5.4-mini # mini + zerank only
    uv run python scripts/listwise_fusion.py --smoke               # biology only
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics

from src.adapters import qab
from src.config import BRIGHT_SUBSETS
from src.domain.fusion import rrf_fuse_rankings
from src.application.listwise import (
    LISTWISE_DIR,
    ListwiseRankings,
    load_listwise_rankings,
    pool_key,
    short_label,
)
from src.domain.metrics import metric as _metric

qab.setup()

METRICS = ("recall_at_1", "recall_at_5", "nDCG_at_10")
METRIC_LABEL = {"recall_at_1": "R@1", "recall_at_5": "R@5", "nDCG_at_10": "nDCG@10"}


def base_label(pool_source: str) -> str:
    # "zerank_only" -> "zerank"; fused pool sources keep their slug.
    return pool_source.removesuffix("_only")


# --------------------------------------------------------------------------- #
# Condition menu                                                               #
# --------------------------------------------------------------------------- #


def build_menu(rankers: list[str]) -> list[dict]:
    """Singletons + every subset of size >= 2, all at EQUAL WEIGHT, RRF only.

    Equal-weight-only is the experiment-wide menu policy (see
    src.conditions): the CE fixed-weight analysis found no tilt beats equal
    beyond noise, so the tilted variants are not tested here either. Fusion
    condition names are member-based (`rrf_<a>+<b>[+<c>...]_equal`) so the
    menu scales past three rankers without ambiguity (the old fixed name
    `rrf_3way_equal` is gone — with base + 3 listwise models the full set is
    a 4-way).
    """
    menu = [{"name": r, "rankers": (r,), "weights": {r: 1.0}} for r in rankers]
    for size in range(2, len(rankers) + 1):
        for combo in itertools.combinations(rankers, size):
            menu.append({
                "name": f"rrf_{'+'.join(combo)}_equal",
                "rankers": combo,
                "weights": {r: 1.0 / size for r in combo},
            })
    return menu


# --------------------------------------------------------------------------- #
# Per-subset evaluation                                                        #
# --------------------------------------------------------------------------- #


def evaluate_subset(
    base: str,
    model_sets: dict[str, ListwiseRankings],
    trials: int,
) -> dict:
    """Evaluate the full menu on one subset.

    Returns {condition: {metric: {trials: [...], mean, std}}} plus extras
    (pool ceiling, agreement, oracle-selector, parse-failure counts).
    """
    any_set = next(iter(model_sets.values()))
    qids = any_set.qids
    gold = any_set.gold
    pool_order = any_set.pool_order

    rankers = [base] + list(model_sets.keys())
    menu = build_menu(rankers)

    def ranking_for(ranker: str, qid: str, trial: int) -> list[str]:
        if ranker == base:
            return pool_order[qid]
        return model_sets[ranker].rankings[qid][trial]

    results: dict[str, dict] = {c["name"]: {m: {"trials": []} for m in METRICS}
                                for c in menu}
    oracle_r1_trials: list[float] = []
    agree1: dict[str, list[float]] = {}

    for t in range(trials):
        per_cond_qvals = {c["name"]: {m: [] for m in METRICS} for c in menu}
        oracle_qvals: list[float] = []
        top1: dict[str, list[str]] = {r: [] for r in rankers}
        for qid in qids:
            g = gold[qid]
            per_ranker = {r: ranking_for(r, qid, t) for r in rankers}
            for r in rankers:
                top1[r].append(per_ranker[r][0])
            for cond in menu:
                if len(cond["rankers"]) == 1:
                    fused = per_ranker[cond["rankers"][0]]
                else:
                    fused = rrf_fuse_rankings(
                        {r: per_ranker[r] for r in cond["rankers"]}, cond["weights"]
                    )
                for m in METRICS:
                    per_cond_qvals[cond["name"]][m].append(_metric(m, g, fused))
            oracle_qvals.append(
                max(_metric("recall_at_1", g, per_ranker[r]) for r in rankers)
            )
        for cond in menu:
            for m in METRICS:
                results[cond["name"]][m]["trials"].append(
                    statistics.fmean(per_cond_qvals[cond["name"]][m])
                )
        oracle_r1_trials.append(statistics.fmean(oracle_qvals))
        for i in range(len(rankers)):
            for j in range(i + 1, len(rankers)):
                a, b = rankers[i], rankers[j]
                frac = statistics.fmean(
                    1.0 if x == y else 0.0 for x, y in zip(top1[a], top1[b])
                )
                agree1.setdefault(f"{a}~{b}", []).append(frac)

    for cond in results.values():
        for m in METRICS:
            ts = cond[m]["trials"]
            cond[m]["mean"] = statistics.fmean(ts)
            cond[m]["std"] = statistics.pstdev(ts) if len(ts) > 1 else 0.0

    pool_ceiling = statistics.fmean(
        _metric(f"recall_at_{any_set.pool_k}", gold[q], pool_order[q]) for q in qids
    )
    parse_failures = {
        label: sum(sum(flags) for flags in ms.parse_failed.values())
        for label, ms in model_sets.items()
    }
    return {
        "n": len(qids),
        "rankers": rankers,
        "conditions": results,
        "pool_ceiling_recall_at_pool_k": pool_ceiling,
        "oracle_selector_r1": {
            "mean": statistics.fmean(oracle_r1_trials),
            "trials": oracle_r1_trials,
        },
        "rank1_agreement": {k: statistics.fmean(v) for k, v in agree1.items()},
        "parse_failures": parse_failures,
    }


# --------------------------------------------------------------------------- #
# Cross-subset report                                                          #
# --------------------------------------------------------------------------- #


def _median_over(subset_payloads: dict[str, dict], cond: str, metric: str,
                 subsets: list[str]) -> float:
    return statistics.median(
        subset_payloads[s]["conditions"][cond][metric]["mean"] for s in subsets
    )


def render_report(payload: dict) -> str:
    subsets = list(payload["per_subset"].keys())
    any_subset = payload["per_subset"][subsets[0]]
    cond_names = list(any_subset["conditions"].keys())
    rankers = any_subset["rankers"]

    lines = [
        "# Listwise fusion (RRF) — " + payload["metadata"]["pool_key"],
        "",
        f"Rankers: {', '.join(rankers)} (base = pool-source CE order; "
        f"models = listwise LLM rerankers, {payload['metadata']['trials']} "
        "trials each, trial-aligned fusion, mean ± std across trials). "
        "RRF k=60. R@pool_k is the pool ceiling (invariant under reordering).",
        "",
    ]
    for m in METRICS:
        lab = METRIC_LABEL[m]
        lines.append(f"## {lab}")
        lines.append("")
        header = "| condition | " + " | ".join(subsets) + " | median |"
        lines.append(header)
        lines.append("|" + "---|" * (len(subsets) + 2))
        for c in cond_names:
            cells = []
            for s in subsets:
                r = payload["per_subset"][s]["conditions"][c][m]
                cells.append(f"{r['mean']:.3f}±{r['std']:.3f}")
            med_all = _median_over(payload["per_subset"], c, m, subsets)
            lines.append(f"| `{c}` | " + " | ".join(cells) +
                         f" | {med_all:.3f} |")
        # Best-fusion-vs-best-singleton line.
        singles = [c for c in cond_names if not c.startswith("rrf_")]
        fusions = [c for c in cond_names if c.startswith("rrf_")]
        if fusions:
            best_s = max(singles, key=lambda c: _median_over(payload["per_subset"], c, m, subsets))
            best_f = max(fusions, key=lambda c: _median_over(payload["per_subset"], c, m, subsets))
            bs = _median_over(payload["per_subset"], best_s, m, subsets)
            bf = _median_over(payload["per_subset"], best_f, m, subsets)
            lines.append("")
            lines.append(
                f"Best singleton (median): `{best_s}` {bs:.3f} · "
                f"best fusion: `{best_f}` {bf:.3f} · fusion lift {bf - bs:+.3f}."
            )
        lines.append("")

    lines.append("## Extras")
    lines.append("")
    lines.append("| subset | n | pool ceiling R@pool_k | oracle-selector R@1 | " +
                 " | ".join(f"agree@1 {k}" for k in any_subset["rank1_agreement"]) +
                 " | parse failures |")
    lines.append("|" + "---|" * (4 + len(any_subset["rank1_agreement"]) + 1))
    for s in subsets:
        p = payload["per_subset"][s]
        agr = " | ".join(f"{v:.3f}" for v in p["rank1_agreement"].values())
        pf = ", ".join(f"{k}:{v}" for k, v in p["parse_failures"].items()) or "0"
        lines.append(
            f"| {s} | {p['n']} | {p['pool_ceiling_recall_at_pool_k']:.3f} | "
            f"{p['oracle_selector_r1']['mean']:.3f} | {agr} | {pf} |"
        )
    lines.append("")
    lines.append("`oracle-selector R@1` = per-query best of the singleton rankers "
                 "(mean over queries, mean over trials) — the routing ceiling at "
                 "this tier. `agree@1` = fraction of queries where two rankers "
                 "pick the same top-1 doc.")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--models", nargs="+",
                    default=["gpt-5.4-mini", "gpt-5.6-luna", "gpt-5.6-terra"],
                    help="Listwise models to fuse (default: the effort-matched "
                         "trio gpt-5.4-mini gpt-5.6-luna gpt-5.6-terra, all "
                         "collected at effort 'none'; pass a subset to analyze "
                         "fewer, or --allow-missing to skip uncollected ones).")
    ap.add_argument("--effort", default="none",
                    help="Reasoning effort the rankings were collected at "
                         "(cache key segment). Default 'none' — the "
                         "experiment setting; the earlier gpt-5.4 caches were "
                         "collected at 'medium'.")
    ap.add_argument("--pool-source", default="zerank_only")
    ap.add_argument("--first-stage-k", type=int, default=2000)
    ap.add_argument("--pool-k", type=int, default=20)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--subsets", nargs="+", default=BRIGHT_SUBSETS)
    ap.add_argument("--allow-missing", action="store_true",
                    help="Proceed with whichever requested models have complete "
                         "caches instead of erroring on the first missing one.")
    ap.add_argument("--smoke", action="store_true", help="Biology only.")
    args = ap.parse_args()
    subsets = ["biology"] if args.smoke else args.subsets

    pk = pool_key(args.pool_source, args.first_stage_k, args.pool_k)
    base = base_label(args.pool_source)
    per_subset: dict[str, dict] = {}
    used_models: list[str] = []

    for domain in subsets:
        model_sets: dict[str, ListwiseRankings] = {}
        for model in args.models:
            try:
                ms = load_listwise_rankings(
                    domain, model, effort=args.effort,
                    pool_source=args.pool_source, first_k=args.first_stage_k,
                    pool_k=args.pool_k, trials=args.trials,
                )
            except (FileNotFoundError, ValueError) as e:
                if not args.allow_missing:
                    raise SystemExit(f"[{domain}] {e}")
                print(f"[{domain}] skipping {model}: {e}")
                continue
            model_sets[short_label(model)] = ms
        if not model_sets:
            raise SystemExit(f"[{domain}] no usable model caches; nothing to fuse.")
        used_models = sorted(set(used_models) | set(model_sets.keys()))
        print(f"=== {domain}: fusing {base} + {sorted(model_sets)} "
              f"({next(iter(model_sets.values())).pool_k}-doc pool, "
              f"{args.trials} trials) ===")
        per_subset[domain] = evaluate_subset(base, model_sets, args.trials)

    payload = {
        "metadata": {
            "pool_key": pk,
            "pool_source": args.pool_source,
            "first_stage_k": args.first_stage_k,
            "pool_k": args.pool_k,
            "models": args.models,
            "model_labels": {m: short_label(m) for m in args.models},
            "effort": args.effort,
            "trials": args.trials,
            "fusion_method": "rrf",
            "rrf_k": 60,
            "trial_policy": "trial-aligned (fuse trial i with trial i; base constant)",
        },
        "per_subset": per_subset,
    }

    out_dir = LISTWISE_DIR / "fusion" / pk
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "fusion.json", "w") as f:
        json.dump(payload, f, indent=1)
    report = render_report(payload)
    (out_dir / "FUSION.md").write_text(report)
    print(f"\nWrote {out_dir}/fusion.json and FUSION.md")
    print("\n" + report)


if __name__ == "__main__":
    main()
