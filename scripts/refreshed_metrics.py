#!/usr/bin/env python3
"""Offline metrics over the refreshed caches (zero API calls).

Derives every menu condition from results/{subdir}/caches/k2000.json and
reports Success@1, nDCG@10, and Recall@20/50/100 per condition, using the
same machinery as the paper runs: DerivedSearchAgent for ranking semantics,
qab's metric functions, and the all-three-present intersection as the query
universe (build_query_set — so robotics' 5+5 voyage/zerank-less queries are
dropped exactly as they were for the June results).

Usage:
    uv run python scripts/refreshed_metrics.py --datasets earth_science robotics
    uv run python scripts/refreshed_metrics.py --datasets earth_science --retrieved-k 200
"""
from __future__ import annotations

import argparse
import json
from statistics import mean

from src.adapters import qab

qab.setup()

from query_agent_benchmarking.internal.adapters.metrics.ir_metrics import (  # noqa: E402
    calculate_nDCG_at_k,
    calculate_recall_at_k,
)

from src.application.derived import DerivedSearchAgent  # noqa: E402
from src.application.queryset import load_and_validate  # noqa: E402
from src.config import DATASETS, DEFAULT_RETRIEVED_K, get_results_dir  # noqa: E402
from src.domain.conditions import build_menu  # noqa: E402

CUTS = (20, 50, 100)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS.keys()),
        default=["earth_science", "robotics"],
    )
    parser.add_argument(
        "--retrieved-k",
        type=int,
        default=DEFAULT_RETRIEVED_K,
        help=f"Hybrid pool prefix fed to each condition (default {DEFAULT_RETRIEVED_K}).",
    )
    args = parser.parse_args()

    for slug in args.datasets:
        print(f"\n=== {slug} (retrieved_k={args.retrieved_k}) ===")
        loaded = load_and_validate(slug)
        if loaded is None:
            continue
        cache, qs = loaded
        golds = {t: sorted(g) for t, g in qs.gold.items()}
        print(f"  query universe: {len(golds)} (all-three intersection)")

        rows = {}
        for condition in build_menu():
            # reranked_k = retrieved_k: rank the whole pool so deep cutoffs
            # are measurable (the paper's reranked_k=20 is an output cap for
            # runs files, not a ranking property).
            agent = DerivedSearchAgent(
                cache,
                retrieved_k=args.retrieved_k,
                condition=condition,
                reranked_k=args.retrieved_k,
            )
            per_metric = {m: [] for m in ("s1", "ndcg10", *[f"r{k}" for k in CUTS])}
            for text, gold in golds.items():
                ranking = agent._derive(text)
                per_metric["s1"].append(calculate_recall_at_k(gold, ranking, 1))
                per_metric["ndcg10"].append(calculate_nDCG_at_k(gold, ranking, 10))
                for k in CUTS:
                    per_metric[f"r{k}"].append(calculate_recall_at_k(gold, ranking, k))
            rows[condition.name] = {m: mean(v) for m, v in per_metric.items()}

        header = f"  {'condition':<22} {'S@1':>7} {'nDCG@10':>8} " + " ".join(
            f"{'R@' + str(k):>7}" for k in CUTS
        )
        print(header)
        for name, r in rows.items():
            print(
                f"  {name:<22} {r['s1']:>7.4f} {r['ndcg10']:>8.4f} "
                + " ".join(f"{r[f'r{k}']:>7.4f}" for k in CUTS)
            )

        out = (
            get_results_dir(slug)
            / "extras"
            / f"refreshed_metrics_k{args.retrieved_k}.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(
                {
                    "retrieved_k": args.retrieved_k,
                    "n_queries": len(golds),
                    "drops": qs.drops,
                    "conditions": rows,
                },
                f,
                indent=2,
            )
        print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
