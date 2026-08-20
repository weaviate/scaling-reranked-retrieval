"""Self-ensemble oracle — the listwise winner's-curse control (S@1 only).

The trio oracle-selector (analysis/listwise_oracle_routing.py) is a per-query
max over 3 stochastic rankings, so part of its headroom is selection-on-noise
(winner's curse), not model diversity. This control isolates that component:
for each query, select the best ranking among ONE model's 3 trials (default
gpt-5.6-luna, the strongest singleton — the conservative control, matching
the CE noise-null's clone-the-strongest-base convention). The self-oracle is
a max over 3 stochastic outputs with ZERO model diversity, so its lift over
the model's own single-trial mean is pure sampling luck. Arm count is matched
by construction: both the trio selector (max over 3 models at one trial) and
the self-oracle (max over 3 trials of one model) maximize over exactly 3
rankings.

Per query:
  - self_oracle(m)   = max over m's trials of S@1        (one number; no
                       trial axis — the 3 trials are consumed by the max)
  - self_bias(m)     = self_oracle(m) − mean-over-trials S@1(m)   (>= 0)
  - oracle_selector  = per-trial max over the models (trial-aligned,
                       identical to listwise_oracle_routing), mean over trials
Derived, the quantities the control exists for:
  - net specialization vs m = oracle_selector − self_oracle(m)
    Positive by a healthy margin => the trio headroom reflects genuine
    per-query specialization beyond what max-over-3-samples manufactures;
    ~0 or negative => the routing ceiling is mostly sampling luck.
  - trial rank-1 self-agreement(m) — mean pairwise fraction of queries where
    two trials of m put the same doc at rank 1 (high agreement => small
    self-oracle lift mechanically; reported to make the bias interpretable).

Metric: Success@1 only (qab's `recall_at_1` hit-rate), matching the routing
analysis. Aggregation: mean across queries per subset (mean ± std across
trials where a trial axis exists) → median across subsets.

Caveat (inline in the report too): the self-oracle taps WITHIN-model sampling
variance while the trio selector taps BETWEEN-model variance at matched arm
count; if a model's trials are more correlated than distinct models are, the
self-oracle is a floor on the winner's-curse component, not an unbiased
estimate of it.

Inputs: the same validated ranking caches as the fusion/routing analyses
(src.listwise.load_listwise_rankings). Zero LLM calls / zero network.

Outputs: results/listwise/self_oracle/<pool-slug>__first<K>__top<P>/
    <labels joined by __>__<effort>.json               full per-subset results
    <labels joined by __>__<effort>__SELF_ORACLE.md    cross-subset report

Usage:
    uv run python scripts/listwise_self_oracle.py            # the trio, luna control
    uv run python scripts/listwise_self_oracle.py --control gpt-5.6-terra
    uv run python scripts/listwise_self_oracle.py --smoke    # biology only
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics

from src.adapters import qab
from src.config import BRIGHT_SUBSETS
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
DEFAULT_CONTROL = "gpt-5.6-luna"


# --------------------------------------------------------------------------- #
# Core                                                                         #
# --------------------------------------------------------------------------- #


def analyze_subset(model_sets: dict[str, ListwiseRankings], trials: int) -> dict:
    """Per-trial singleton + trio-selector S@1 means, plus per-model
    self-oracle / self-bias / trial rank-1 self-agreement (no trial axis —
    the trials are consumed by the max / the pairwise comparison)."""
    labels = list(model_sets)
    sets_ = list(model_sets.values())
    if any(ms.qids != sets_[0].qids for ms in sets_[1:]):
        raise ValueError("Model caches cover different query sets — same pool "
                         "expected for a self-oracle control.")
    qids = sets_[0].qids
    gold = sets_[0].gold
    n = len(qids)

    # s1[m][qid] = per-trial S@1 list for model m on query qid.
    s1 = {
        m: {qid: [_metric(METRIC, gold[qid], model_sets[m].rankings[qid][t])
                  for t in range(trials)]
            for qid in qids}
        for m in labels
    }

    values: dict[str, dict] = {}
    for m in labels:
        per_trial = [statistics.fmean(s1[m][qid][t] for qid in qids)
                     for t in range(trials)]
        values[m] = {"trials": per_trial, "mean": statistics.fmean(per_trial),
                     "std": statistics.pstdev(per_trial)}
        self_oracle = statistics.fmean(max(s1[m][qid]) for qid in qids)
        values[f"self_oracle:{m}"] = {"mean": self_oracle, "std": 0.0}
        values[f"self_bias:{m}"] = {
            "mean": self_oracle - values[m]["mean"], "std": 0.0}
        pairs = list(itertools.combinations(range(trials), 2))
        agree = statistics.fmean(
            sum(model_sets[m].rankings[qid][a][0] == model_sets[m].rankings[qid][b][0]
                for qid in qids) / n
            for a, b in pairs)
        values[f"trial_agreement:{m}"] = {"mean": agree, "std": 0.0}

    sel_trials = [
        statistics.fmean(max(s1[m][qid][t] for m in labels) for qid in qids)
        for t in range(trials)
    ]
    values["oracle_selector"] = {
        "trials": sel_trials, "mean": statistics.fmean(sel_trials),
        "std": statistics.pstdev(sel_trials)}

    return {"n": n, "labels": labels, "values": values}


def summarize(payload: dict, control: str) -> None:
    """Attach the derived comparison lines per subset: net specialization vs
    each model's self-oracle, trio selection headroom, best singleton."""
    for p in payload["per_subset"].values():
        v = p["values"]
        labels = p["labels"]
        sel = v["oracle_selector"]["mean"]
        best_singleton = max(labels, key=lambda m: v[m]["mean"])
        p["best_singleton"] = {"which": best_singleton,
                               "mean": v[best_singleton]["mean"]}
        v["selection_headroom"] = {
            "mean": sel - v[best_singleton]["mean"], "std": 0.0}
        for m in labels:
            v[f"net_specialization:{m}"] = {
                "mean": sel - v[f"self_oracle:{m}"]["mean"], "std": 0.0}
        max_self = max(labels, key=lambda m: v[f"self_oracle:{m}"]["mean"])
        p["max_self_oracle"] = {"which": max_self,
                                "mean": v[f"self_oracle:{max_self}"]["mean"]}
        v["net_specialization:max_self"] = {
            "mean": sel - v[f"self_oracle:{max_self}"]["mean"], "std": 0.0}
    payload["metadata"]["control"] = control


# --------------------------------------------------------------------------- #
# Report                                                                       #
# --------------------------------------------------------------------------- #


def _median(per_subset: dict, key: str, subsets: list[str]) -> float:
    return statistics.median(per_subset[s]["values"][key]["mean"] for s in subsets)


def render_report(payload: dict) -> str:
    md = payload["metadata"]
    labels = md["labels"]
    control = md["control_label"]
    subsets = list(payload["per_subset"].keys())
    per_subset = payload["per_subset"]

    rows: list[tuple[str, str, bool]] = [(m, f"`{m}` (single-trial mean)", False)
                                         for m in labels]
    for m in labels:
        rows.append((f"self_oracle:{m}",
                     f"self-oracle `{m}` (per-query best of {md['trials']} "
                     "trials)", False))
    rows += [
        ("oracle_selector",
         "trio oracle-selector (per-query best model, trial-aligned)", False),
        (f"self_bias:{control}",
         f"**self-bias `{control}`** (self-oracle − single-trial mean; "
         "pure winner's curse)", True),
        ("selection_headroom",
         "trio selection headroom (selector − best singleton)", True),
        (f"net_specialization:{control}",
         f"**net specialization** (selector − `{control}` self-oracle)", True),
        ("net_specialization:max_self",
         "net specialization vs max self-oracle (most conservative)", True),
    ]

    lines = [
        f"# Listwise self-ensemble oracle control — {' vs '.join(labels)} — "
        f"{md['pool_key']} — S@1",
        "",
        "Winner's-curse control for the trio oracle-selector: per query, "
        f"select the best of ONE model's {md['trials']} trials (`{control}`, "
        "the strongest singleton — conservative, matching the CE noise-null's "
        "clone-the-strongest-base convention). The self-oracle is a max over "
        f"{md['trials']} stochastic rankings with zero model diversity, so its "
        "lift over the single-trial mean is manufactured entirely by "
        "selection-on-noise. Arm count is matched: the trio selector is also "
        f"a max over {len(labels)} rankings. Metric: **Success@1** (qab's "
        "`recall_at_1` hit-rate). Medians are across subsets of per-subset "
        "means; ± is across-trial std where a trial axis exists.",
        "",
        "| line | " + " | ".join(subsets) +
        " | median |",
        "|" + "---|" * (len(subsets) + 2),
    ]
    for key, label, signed in rows:
        cells = []
        for s in subsets:
            v = per_subset[s]["values"][key]
            if v.get("trials"):
                cells.append(f"{v['mean']:.3f}±{v['std']:.3f}")
            else:
                fmt_c = "+.3f" if signed else ".3f"
                cells.append(f"{v['mean']:{fmt_c}}")
        med_all = _median(per_subset, key, subsets)
        fmt = "+.3f" if signed else ".3f"
        lines.append(f"| {label} | " + " | ".join(cells) +
                     f" | {med_all:{fmt}} |")

    lines += [
        "",
        "Trial rank-1 self-agreement (mean pairwise fraction of queries where "
        "two trials place the same doc at rank 1 — high agreement mechanically "
        "caps the self-oracle lift): " + "; ".join(
            f"`{m}` " + ", ".join(
                f"{s} {per_subset[s]['values'][f'trial_agreement:{m}']['mean']:.2f}"
                for s in subsets)
            for m in labels) + ".",
        "",
        "`max self-oracle` picks each subset's largest self-oracle among the "
        "models: " + ", ".join(
            f"{s} = `{per_subset[s]['max_self_oracle']['which']}`"
            for s in subsets) + ".",
        "",
        "**Reading:** if net specialization is clearly positive, the trio "
        "selector's headroom is not reproducible by resampling one model — "
        "it reflects genuine per-query model specialization. If it is ~0 or "
        "negative, the routing ceiling is mostly sampling luck. **Caveat:** "
        "the self-oracle taps within-model sampling variance while the trio "
        "selector taps between-model variance; if one model's trials are more "
        "correlated than distinct models are, the self-oracle is a *floor* on "
        "the winner's-curse component, not an unbiased estimate of it.",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help="Listwise models in the trio selector (default: the "
                         "effort-matched trio).")
    ap.add_argument("--control", default=DEFAULT_CONTROL,
                    help="Model whose self-oracle is the headline control "
                         "(default gpt-5.6-luna; self-oracles are computed "
                         "for every model regardless).")
    ap.add_argument("--effort", default="none")
    ap.add_argument("--pool-source", default="zerank_only")
    ap.add_argument("--first-stage-k", type=int, default=2000)
    ap.add_argument("--pool-k", type=int, default=20)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--subsets", nargs="+", default=BRIGHT_SUBSETS)
    ap.add_argument("--smoke", action="store_true", help="Biology only.")
    args = ap.parse_args()
    if len(args.models) < 2:
        raise SystemExit("Need at least two --models for a trio selector.")
    if args.control not in args.models:
        raise SystemExit(f"--control {args.control} must be one of --models.")
    subsets = ["biology"] if args.smoke else args.subsets

    labels = [short_label(m) for m in args.models]
    control_label = short_label(args.control)
    pk = pool_key(args.pool_source, args.first_stage_k, args.pool_k)

    payload = {
        "metadata": {
            "pool_key": pk,
            "pool_source": args.pool_source,
            "first_stage_k": args.first_stage_k,
            "pool_k": args.pool_k,
            "models": list(args.models),
            "labels": labels,
            "control_label": control_label,
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

    summarize(payload, args.control)
    for domain, p in payload["per_subset"].items():
        v = p["values"]
        print(f"[{domain}] n={p['n']}: "
              + " / ".join(f"{m} {v[m]['mean']:.3f}" for m in labels)
              + f" / self[{control_label}] "
              f"{v[f'self_oracle:{control_label}']['mean']:.3f}"
              f" / selector {v['oracle_selector']['mean']:.3f}"
              f" (self-bias {v[f'self_bias:{control_label}']['mean']:+.3f}, "
              f"net spec {v[f'net_specialization:{control_label}']['mean']:+.3f})")

    out_dir = LISTWISE_DIR / "self_oracle" / pk
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "__".join(labels) + f"__{args.effort}"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}__SELF_ORACLE.md"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=1)
    md_path.write_text(render_report(payload))
    print(f"\nWrote {json_path}\nWrote {md_path}")


if __name__ == "__main__":
    main()
