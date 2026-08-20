"""Per-model unique successes among the listwise LLM rerankers.

The CE-tier unique-success analysis (analysis/unique_successes.py), ported to
the listwise tier over an arbitrary model set (default: the effort-`none`
trio gpt-5.4-mini, gpt-5.6-luna, gpt-5.6-terra). A model has a UNIQUE SUCCESS
on a query (at cutoff K) when it lands a gold doc in its top-K while EVERY
other model in the set misses — the singleton cells of the Success@K Venn.
This is the per-model heterogeneity a router feeds on: the queries each
listwise model alone rescues. The pool-source CE baseline is NOT a member;
the Venn is over the listwise models only. With two models this reduces
exactly to the pairwise a-only / b-only / both / neither analysis.

"Success at K" = recall@K > 0 (at least one gold in the top-K), matching the
CE analysis. Cutoffs are @1 and @5 ONLY: all models permute the SAME fixed
pool (the pool-source top-pool_k), so hits at @pool_k are identical by
construction and no unique success is possible there.

Trials: listwise LLM sampling is stochastic, so the Venn is computed per
TRIAL-ALIGNED trial (trial t of every model — the fusion analysis' policy)
and cells are reported as mean ± std across trials. Rescued-query lists keep
only queries uniquely rescued in a MAJORITY of trials (stable rescues), with
the per-query trial count retained (full per-trial lists in the JSON).

Cells per cutoff: one `unique` count per model, `multi_hit` (>= 2 models
hit), `all_miss` (no model hits). unique counts + multi_hit + all_miss = n.

Inputs (all on disk; zero LLM / zero network):
  - results/listwise/pools/<domain>__<pool-slug>__first<K>__top<P>.json
  - results/listwise/cache/<domain>__<model>__<effort>__<pool-slug>__...jsonl
    (loaded + validated via src.listwise.load_listwise_rankings)

Outputs: results/listwise/unique_successes/<pool-slug>__first<K>__top<P>/
    <labels joined by __>__<effort>.json          full per-subset results
    <labels joined by __>__<effort>__UNIQUE.md    cross-subset report

Usage:
    uv run python scripts/listwise_unique_successes.py           # the trio
    uv run python scripts/listwise_unique_successes.py \
        --models gpt-5.6-luna gpt-5.6-terra                       # any subset
    uv run python scripts/listwise_unique_successes.py --smoke   # biology only
"""
from __future__ import annotations

import argparse
import json
import statistics

from src.adapters import qab
from src.config import BRIGHT_SUBSETS
from src.application.listwise import (
    LISTWISE_DIR,
    ListwiseRankings,
    load_listwise_rankings,
    load_pool,
    pool_key,
    short_label,
)
from src.domain.metrics import metric as _metric

qab.setup()

SUCCESS_CUTOFFS = (1, 5)  # @pool_k excluded: identical pools => no uniques.
QUERY_TRUNC = 120

DEFAULT_MODELS = ["gpt-5.4-mini", "gpt-5.6-luna", "gpt-5.6-terra"]


# --------------------------------------------------------------------------- #
# Core                                                                         #
# --------------------------------------------------------------------------- #


def qid_to_query_text(domain: str, pool_source: str, first_k: int,
                      pool_k: int) -> dict[str, str]:
    """qid -> (truncated) query text, from the materialized pool file."""
    pool = load_pool(domain, pool_source, first_k, pool_k)
    return {q["qid"]: text[:QUERY_TRUNC] for text, q in pool["queries"].items()}


def analyze_subset(model_sets: dict[str, ListwiseRankings], trials: int) -> dict:
    """Per-(cutoff, trial) Venn cells + per-query unique-rescue counts.

    model_sets is keyed by short label. Returns per cutoff: per-trial cell
    counts, mean/std per cell, and per model the qids it uniquely rescued
    with the number of trials (out of `trials`) the rescue held in.
    """
    labels = list(model_sets)
    sets_ = list(model_sets.values())
    if any(ms.qids != sets_[0].qids for ms in sets_[1:]):
        raise ValueError("Model caches cover different query sets — same pool "
                         "expected for a Venn.")
    qids = sets_[0].qids
    gold = sets_[0].gold

    shared_cells = ("multi_hit", "all_miss")
    per_trial = {K: {**{m: [] for m in labels},
                     **{c: [] for c in shared_cells}}
                 for K in SUCCESS_CUTOFFS}
    uniq_count = {K: {m: {} for m in labels} for K in SUCCESS_CUTOFFS}

    for t in range(trials):
        counts = {K: dict.fromkeys(list(labels) + list(shared_cells), 0)
                  for K in SUCCESS_CUTOFFS}
        for qid in qids:
            g = gold[qid]
            rankings = {m: model_sets[m].rankings[qid][t] for m in labels}
            for K in SUCCESS_CUTOFFS:
                hits = [m for m in labels
                        if _metric(f"recall_at_{K}", g, rankings[m]) > 0]
                if len(hits) == 0:
                    counts[K]["all_miss"] += 1
                elif len(hits) == 1:
                    m = hits[0]
                    counts[K][m] += 1
                    uniq_count[K][m][qid] = uniq_count[K][m].get(qid, 0) + 1
                else:
                    counts[K]["multi_hit"] += 1
        for K in SUCCESS_CUTOFFS:
            for c in counts[K]:
                per_trial[K][c].append(counts[K][c])

    majority = trials // 2 + 1
    out: dict[str, dict] = {}
    for K in SUCCESS_CUTOFFS:
        stats = {}
        for c, ts in per_trial[K].items():
            stats[c] = {
                "trials": ts,
                "mean": statistics.fmean(ts),
                "std": statistics.pstdev(ts) if len(ts) > 1 else 0.0,
            }
        stats["unique_queries"] = {
            m: {
                "all": dict(sorted(uniq_count[K][m].items())),
                "stable": sorted(
                    q for q, n in uniq_count[K][m].items() if n >= majority
                ),
            }
            for m in labels
        }
        out[str(K)] = stats
    return {"n": len(qids), "labels": labels, "cutoffs": out,
            "majority_trials": majority}


# --------------------------------------------------------------------------- #
# Report                                                                       #
# --------------------------------------------------------------------------- #


def render_report(payload: dict, list_cutoff: int) -> str:
    md = payload["metadata"]
    labels = md["labels"]
    subsets = list(payload["per_subset"].keys())
    trials = md["trials"]
    cols = list(labels) + ["multi_hit", "all_miss"]
    col_label = {**{m: f"{m}-only" for m in labels},
                 "multi_hit": "≥2 hit", "all_miss": "all miss"}

    lines = [
        f"# Listwise unique successes — {' vs '.join(labels)} — {md['pool_key']}",
        "",
        "Models: " + ", ".join(f"`{mod}` ({lab})" for mod, lab
                               in zip(md["models"], labels))
        + f", effort {md['effort']}, over the {md['pool_source']} "
        f"top-{md['pool_k']} pool. A unique success @K = a gold doc in one "
        "model's top-K while every other model misses (Success@K = "
        f"recall@K > 0). Trial-aligned over {trials} trials; cells are mean "
        f"± std across trials. @{md['pool_k']} is omitted: all models "
        "permute the same pool, so no unique success is possible there.",
        "",
        "## Counts (mean ± std across trials)",
        "",
    ]
    header = "| subset | n |"
    for K in SUCCESS_CUTOFFS:
        for c in cols:
            header += f" @{K} {col_label[c]} |"
    lines.append(header)
    lines.append("|" + "---|" * (2 + len(cols) * len(SUCCESS_CUTOFFS)))

    totals = {K: dict.fromkeys(cols, 0.0) for K in SUCCESS_CUTOFFS}
    total_n = 0
    for s in subsets:
        p = payload["per_subset"][s]
        row = f"| {s} | {p['n']} |"
        total_n += p["n"]
        for K in SUCCESS_CUTOFFS:
            st = p["cutoffs"][str(K)]
            for c in cols:
                row += f" {st[c]['mean']:.1f}±{st[c]['std']:.1f} |"
                totals[K][c] += st[c]["mean"]
        lines.append(row)
    row = f"| **total** | **{total_n}** |"
    for K in SUCCESS_CUTOFFS:
        for c in cols:
            row += f" **{totals[K][c]:.1f}** |"
    lines.append(row)
    lines.append("")
    for K in SUCCESS_CUTOFFS:
        lines.append(
            f"Unique-success rates on the {total_n}-query total @{K}: "
            + ", ".join(f"{m} {totals[K][m] / total_n:.3f}" for m in labels)
            + "."
        )
    lines.append("")

    K = list_cutoff
    majority = payload["per_subset"][subsets[0]]["majority_trials"]
    lines.append(f"## Stable rescued queries @{K} "
                 f"(uniquely rescued in ≥{majority}/{trials} trials)")
    lines.append("")
    for m in labels:
        lines.append(f"### {m}")
        lines.append("")
        any_listed = False
        for s in subsets:
            p = payload["per_subset"][s]
            uq = p["cutoffs"][str(K)]["unique_queries"][m]
            texts = payload["query_texts"].get(s, {})
            for qid in uq["stable"]:
                n_tr = uq["all"][qid]
                txt = texts.get(qid, "")
                lines.append(f"- {s} · q{qid} ({n_tr}/{trials}): {txt}")
                any_listed = True
        if not any_listed:
            lines.append("- (none)")
        lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help="Listwise models in the Venn (default: the "
                         "effort-matched trio; any two give the pairwise "
                         "a-only/b-only analysis).")
    ap.add_argument("--effort", default="none",
                    help="Reasoning effort the rankings were collected at "
                         "(cache key segment; default 'none').")
    ap.add_argument("--pool-source", default="zerank_only")
    ap.add_argument("--first-stage-k", type=int, default=2000)
    ap.add_argument("--pool-k", type=int, default=20)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--subsets", nargs="+", default=BRIGHT_SUBSETS)
    ap.add_argument("--list-cutoff", type=int, default=1,
                    choices=list(SUCCESS_CUTOFFS),
                    help="Cutoff whose stable rescued queries are listed in "
                         "the markdown report (default 1; all cutoffs are in "
                         "the JSON).")
    ap.add_argument("--smoke", action="store_true", help="Biology only.")
    args = ap.parse_args()
    if len(args.models) < 2:
        raise SystemExit("Need at least two --models for a Venn.")
    subsets = ["biology"] if args.smoke else args.subsets

    labels = [short_label(m) for m in args.models]
    pk = pool_key(args.pool_source, args.first_stage_k, args.pool_k)

    per_subset: dict[str, dict] = {}
    query_texts: dict[str, dict[str, str]] = {}
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
        per_subset[domain] = analyze_subset(ms, args.trials)
        query_texts[domain] = qid_to_query_text(
            domain, args.pool_source, args.first_stage_k, args.pool_k
        )
        print(f"[{domain}] n={per_subset[domain]['n']}: "
              + "; ".join(
                  f"@{K} " + " / ".join(
                      f"{m} {per_subset[domain]['cutoffs'][str(K)][m]['mean']:.1f}"
                      for m in labels)
                  for K in SUCCESS_CUTOFFS))

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
            "success_cutoffs": list(SUCCESS_CUTOFFS),
        },
        "per_subset": per_subset,
        "query_texts": query_texts,
    }

    out_dir = LISTWISE_DIR / "unique_successes" / pk
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "__".join(labels) + f"__{args.effort}"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}__UNIQUE.md"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=1)
    md_path.write_text(render_report(payload, args.list_cutoff))
    print(f"\nWrote {json_path}\nWrote {md_path}")


if __name__ == "__main__":
    main()
