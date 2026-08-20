"""Oracle routing (binary selection) among the listwise models — S@1 only.

The CE-tier routing analysis (analysis/oracle_config.py), ported to the
listwise tier over an arbitrary model set (default: the effort-`none` trio
gpt-5.4-mini, gpt-5.6-luna, gpt-5.6-terra; the pool-source CE baseline is
NOT a menu member). Metric is Success@1 ONLY — qab's `recall_at_1` is a
hit-rate (a gold doc at rank 1), i.e. Success@1; this analysis uses the
honest name.

SELECTION-ONLY BY DESIGN: the object of interest is the per-query BINARY
router (pick one model per query), so the oracle here is `oracle_selector`
= per-query best of the model singletons. The weighted-oracle quantities of
the CE decomposition (`oracle_config`, `blending value` — per-query best
over singletons + blends) are deliberately NOT computed: per-query weighted
fusion is a different object with a worse winner's-curse exposure (the CE
paper's "object 3", left as a direction there too).

Per trial (trial-aligned, matching analysis/listwise_fusion.py), per query:
  - S@1 of each singleton
  - `best_static_fusion` — ONE fixed equal-weight RRF blend applied to every
    query: the argmax over all equal-weight blends of the models (every
    subset of size >= 2, RRF k=60) by pooled-mean S@1 across all subsets ×
    queries × trials (the listwise analog of the CE decomposition's
    "one fixed blend, re-selected by pooled-mean"). The chosen blend is
    named in the output.
  - `oracle_selector` = per-query best of the model singletons
Derived, the paper §5.1 vocabulary:
  - routing value       = oracle_selector − best_static_fusion
  - selection headroom  = oracle_selector − best singleton (winner's-curse-
    exposed; >= 0 by construction)

Aggregation: mean across queries per (subset, trial) → mean ± std across
trials per subset → median across subsets.

Winner's-curse caveat (reported inline, not buried): the per-query max over
N stochastic rankers is upward-biased under noise — the exposure the CE
tier's noise-null (§5.1) quantifies, and it grows with N. Oracle lines are
ceilings for a learned router, not achieved results; no listwise noise-null
has been run.

Inputs: the same validated ranking caches as the fusion analysis
(src.listwise.load_listwise_rankings). Zero LLM calls / zero network.

Outputs: results/listwise/oracle_routing/<pool-slug>__first<K>__top<P>/
    <labels joined by __>__<effort>.json           full per-subset results
    <labels joined by __>__<effort>__ROUTING.md    cross-subset report

Usage:
    uv run python scripts/listwise_oracle_routing.py            # the trio
    uv run python scripts/listwise_oracle_routing.py \
        --models gpt-5.6-luna gpt-5.6-terra                      # any subset
    uv run python scripts/listwise_oracle_routing.py --smoke    # biology only
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

METRIC = "recall_at_1"  # qab's name for the S@1 hit-rate; reported as S@1.

DEFAULT_MODELS = ["gpt-5.4-mini", "gpt-5.6-luna", "gpt-5.6-terra"]


# --------------------------------------------------------------------------- #
# Core                                                                         #
# --------------------------------------------------------------------------- #


def blend_menu(labels: list[str]) -> list[tuple[str, tuple[str, ...]]]:
    """Every equal-weight RRF blend of the models: (name, members), size >= 2."""
    out = []
    for size in range(2, len(labels) + 1):
        for combo in itertools.combinations(labels, size):
            out.append((f"rrf_{'+'.join(combo)}_equal", combo))
    return out


def analyze_subset(model_sets: dict[str, ListwiseRankings], trials: int) -> dict:
    """Per-trial S@1 means for every singleton, every blend, and the
    per-query selector (max over singletons). Blend selection happens later,
    globally, from the pooled means."""
    labels = list(model_sets)
    sets_ = list(model_sets.values())
    if any(ms.qids != sets_[0].qids for ms in sets_[1:]):
        raise ValueError("Model caches cover different query sets — same pool "
                         "expected for a routing decomposition.")
    qids = sets_[0].qids
    gold = sets_[0].gold
    blends = blend_menu(labels)

    keys = list(labels) + [name for name, _ in blends] + ["oracle_selector"]
    per_trial: dict[str, list[float]] = {k: [] for k in keys}
    for t in range(trials):
        sums = dict.fromkeys(keys, 0.0)
        for qid in qids:
            g = gold[qid]
            rankings = {m: model_sets[m].rankings[qid][t] for m in labels}
            s = {m: _metric(METRIC, g, rankings[m]) for m in labels}
            for m in labels:
                sums[m] += s[m]
            sums["oracle_selector"] += max(s.values())
            for name, members in blends:
                fused = rrf_fuse_rankings(
                    {m: rankings[m] for m in members},
                    {m: 1.0 / len(members) for m in members},
                )
                sums[name] += _metric(METRIC, g, fused)
        n = len(qids)
        for k in keys:
            per_trial[k].append(sums[k] / n)

    out: dict[str, dict] = {}
    for k in keys:
        ts = per_trial[k]
        out[k] = {"trials": ts, "mean": statistics.fmean(ts),
                  "std": statistics.pstdev(ts) if len(ts) > 1 else 0.0}
    return {"n": len(qids), "labels": labels,
            "blend_names": [name for name, _ in blends], "values": out}


def summarize(payload: dict) -> None:
    """Pick the global best static blend (pooled-mean S@1 across all subsets ×
    queries × trials), then attach the derived per-subset lines:
    best_static_fusion, routing_value, selection_headroom, best_singleton."""
    per_subset = payload["per_subset"]
    any_sub = next(iter(per_subset.values()))
    labels = any_sub["labels"]
    blend_names = any_sub["blend_names"]

    def pooled_mean(key: str) -> float:
        num = sum(p["values"][key]["mean"] * p["n"] for p in per_subset.values())
        den = sum(p["n"] for p in per_subset.values())
        return num / den

    best_blend = max(blend_names, key=pooled_mean)
    payload["metadata"]["best_static_fusion"] = {
        "name": best_blend,
        "pooled_mean_s1": pooled_mean(best_blend),
        "pooled_mean_s1_all_blends": {b: pooled_mean(b) for b in blend_names},
        "note": ("One fixed equal-weight blend applied to every query/subset; "
                 "argmax by pooled-mean S@1 over all subsets × queries × "
                 "trials."),
    }

    for p in per_subset.values():
        v = p["values"]
        best_singleton = max(labels, key=lambda m: v[m]["mean"])
        sel, blend = v["oracle_selector"]["trials"], v[best_blend]["trials"]
        best = v[best_singleton]["trials"]
        for key, ts in (
            ("routing_value", [s - f for s, f in zip(sel, blend)]),
            ("selection_headroom", [s - x for s, x in zip(sel, best)]),
        ):
            v[key] = {"trials": ts, "mean": statistics.fmean(ts),
                      "std": statistics.pstdev(ts) if len(ts) > 1 else 0.0}
        p["best_singleton"] = {"which": best_singleton,
                               "mean": v[best_singleton]["mean"]}


# --------------------------------------------------------------------------- #
# Report                                                                       #
# --------------------------------------------------------------------------- #


def _median(per_subset: dict, key: str, subsets: list[str]) -> float:
    return statistics.median(per_subset[s]["values"][key]["mean"] for s in subsets)


def render_report(payload: dict) -> str:
    md = payload["metadata"]
    labels = md["labels"]
    best_blend = md["best_static_fusion"]["name"]
    subsets = list(payload["per_subset"].keys())
    per_subset = payload["per_subset"]

    rows = [(m, f"`{m}`", False) for m in labels] + [
        (best_blend, f"`{best_blend}` (best static fusion)", False),
        ("oracle_selector",
         "oracle-selector (per-query best singleton — binary selection)", False),
        ("routing_value", "**routing value** (selector − static fusion)", True),
        ("selection_headroom",
         "selection headroom (selector − best singleton)", True),
    ]

    lines = [
        f"# Listwise oracle routing — {' vs '.join(labels)} — "
        f"{md['pool_key']} — S@1",
        "",
        "Binary-selection routing analysis over the listwise models only ("
        + ", ".join(f"`{mod}` ({lab})" for mod, lab in zip(md["models"], labels))
        + f", effort {md['effort']}; the {md['pool_source']} CE baseline is "
        "not a menu member). Metric: **Success@1** — a gold doc at rank 1 "
        "(qab's `recall_at_1` hit-rate). The oracle is the per-query best "
        "SINGLETON (pick one model per query); per-query weighted fusion "
        "(`oracle_config` / blending value) is deliberately not computed. "
        f"Trial-aligned over {md['trials']} trials; cells are mean ± std "
        "across trials; medians are across subsets of per-subset means. "
        f"Best static fusion = `{best_blend}` (argmax by pooled-mean S@1 "
        "over all equal-weight blends; per-blend pooled means in the JSON).",
        "",
        "| line | " + " | ".join(subsets) +
        " | median |",
        "|" + "---|" * (len(subsets) + 2),
    ]
    for key, label, signed in rows:
        cells = []
        for s in subsets:
            v = per_subset[s]["values"][key]
            cells.append(f"{v['mean']:.3f}±{v['std']:.3f}")
        med_all = _median(per_subset, key, subsets)
        fmt = "+.3f" if signed else ".3f"
        lines.append(f"| {label} | " + " | ".join(cells) +
                     f" | {med_all:{fmt}} |")
    lines += [
        "",
        "`selection headroom` uses each subset's own best singleton (by "
        "trial-mean): " + ", ".join(
            f"{s} = `{per_subset[s]['best_singleton']['which']}`"
            for s in subsets) + ".",
        "",
        f"**Winner's-curse caveat:** `oracle-selector` is a per-query max "
        f"over {len(labels)} stochastic rankers and is upward-biased under "
        "noise — the exposure the CE tier's noise-null (§5.1) quantifies, "
        "and it grows with the number of arms. Read it as a ceiling for a "
        "learned router, not an achieved result; no listwise noise-null has "
        "been run.",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help="Listwise models in the routing menu (default: the "
                         "effort-matched trio).")
    ap.add_argument("--effort", default="none")
    ap.add_argument("--pool-source", default="zerank_only")
    ap.add_argument("--first-stage-k", type=int, default=2000)
    ap.add_argument("--pool-k", type=int, default=20)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--subsets", nargs="+", default=BRIGHT_SUBSETS)
    ap.add_argument("--smoke", action="store_true", help="Biology only.")
    args = ap.parse_args()
    if len(args.models) < 2:
        raise SystemExit("Need at least two --models for a routing menu.")
    subsets = ["biology"] if args.smoke else args.subsets

    labels = [short_label(m) for m in args.models]
    pk = pool_key(args.pool_source, args.first_stage_k, args.pool_k)

    payload = {
        "metadata": {
            "pool_key": pk,
            "pool_source": args.pool_source,
            "first_stage_k": args.first_stage_k,
            "pool_k": args.pool_k,
            "models": list(args.models),
            "labels": labels,
            "effort": args.effort,
            "trials": args.trials,
            "metric": "success_at_1 (qab recall_at_1 hit-rate)",
        },
        "per_subset": {},
    }
    for domain in subsets:
        ms: dict[str, ListwiseRankings] = {}
        for model, label in zip(args.models, labels):
            try:
                ms[label] = load_listwise_rankings(
                    domain, model, effort=args.effort,
                    pool_source=args.pool_source, first_k=args.first_stage_k,
                    pool_k=args.pool_k, trials=args.trials,
                )
            except (FileNotFoundError, ValueError) as e:
                raise SystemExit(f"[{domain}] {e}")
        payload["per_subset"][domain] = analyze_subset(ms, args.trials)

    summarize(payload)
    best_blend = payload["metadata"]["best_static_fusion"]["name"]
    for domain, p in payload["per_subset"].items():
        v = p["values"]
        print(f"[{domain}] n={p['n']}: "
              + " / ".join(f"{m} {v[m]['mean']:.3f}" for m in labels)
              + f" / fused[{best_blend}] {v[best_blend]['mean']:.3f}"
              f" / selector {v['oracle_selector']['mean']:.3f}"
              f" (routing {v['routing_value']['mean']:+.3f}, headroom "
              f"{v['selection_headroom']['mean']:+.3f})")

    out_dir = LISTWISE_DIR / "oracle_routing" / pk
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "__".join(labels) + f"__{args.effort}"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}__ROUTING.md"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=1)
    md_path.write_text(render_report(payload))
    print(f"\nWrote {json_path}\nWrote {md_path}")


if __name__ == "__main__":
    main()
