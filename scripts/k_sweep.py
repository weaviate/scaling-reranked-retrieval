#!/usr/bin/env python3
"""Derive results at every retrieved_k from one collected cache (zero API calls).

Run script — loops scripts/run_experiment.py --from-cache over
k in {100, 200, 500, 1000, 2000} against the dataset's k=2000 cache
(collect it first: uv run python scripts/run_experiment.py --dataset <slug>
--retrieved-k 2000 --collect-only).

Usage:
    uv run python scripts/k_sweep.py <dataset> [--reranked-k 100] [--cache-k 2000]

Each derived run writes results/<subdir>/runs/k{N}_from_k{M}.json
(runs_rk{R}/ for a non-default --reranked-k, e.g. 100 to measure R@50/R@100).
"""
import argparse
import subprocess
import sys
from pathlib import Path

from src.config import DATASETS, RESULTS_DIR

SWEEP_KS = (100, 200, 500, 1000, 2000)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(DATASETS.keys()))
    parser.add_argument(
        "--reranked-k", type=int, default=20,
        help="Post-rerank output cap (default 20; 100 enables R@50/R@100 and "
             "routes output to runs_rk100/).",
    )
    parser.add_argument(
        "--cache-k", type=int, default=2000,
        help="Pool size of the collected cache to derive from (default 2000).",
    )
    args = parser.parse_args()

    subdir = DATASETS[args.dataset].results_subdir or args.dataset
    cache = RESULTS_DIR / subdir / "caches" / f"k{args.cache_k}.json"
    if not cache.exists():
        raise SystemExit(
            f"Cache not found: {cache}\nCollect it first:\n"
            f"  uv run python scripts/run_experiment.py --dataset {args.dataset} "
            f"--retrieved-k {args.cache_k} --collect-only"
        )

    run_experiment = Path(__file__).with_name("run_experiment.py")
    for k in (n for n in SWEEP_KS if n <= args.cache_k):
        cmd = [
            sys.executable, str(run_experiment),
            "--dataset", args.dataset,
            "--retrieved-k", str(k),
            "--reranked-k", str(args.reranked_k),
            "--from-cache", str(cache),
        ]
        print(f"\n=== k={k} (reranked_k={args.reranked_k}) ===")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
