#!/usr/bin/env python3
"""First-stage hybrid-only eval: recall at retrieved_k (LIVE Weaviate calls).

Reproduces the hybrid_only row of run_experiment without requiring reranker
API keys: one Weaviate hybrid search per query (BM25 + arctic embeddings,
relative-score fusion, alpha 0.75 — the BaseRetriever defaults used by every
paper run) at --retrieved-k, scored by qab against the dataset's gold ids.
Reports the profile metrics (recall@1/5/20, nDCG@10) plus the deep recall
cutoffs <= retrieved_k.

Usage:
    uv run python scripts/hybrid_recall.py --dataset robotics --retrieved-k 2000
"""
from __future__ import annotations

import argparse
import json
import os

from src.adapters import qab

qab.setup()

from query_agent_benchmarking import run_search_eval  # noqa: E402

from src.adapters.qab import RetrieverSearchAgent  # noqa: E402
from src.adapters.retrieval.base_retriever import BaseRetriever  # noqa: E402
from src.config import DATASETS, RANDOM_SEED, get_results_dir  # noqa: E402
from src.domain.metrics import build_extra_metrics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS.keys()),
        required=True,
        help="Dataset slug from src.config.DATASETS.",
    )
    parser.add_argument(
        "--retrieved-k",
        type=int,
        default=2000,
        help="Hybrid pool size returned per query. Default: 2000.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit number of queries (default: full dataset).",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Use sync execution instead of async.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        help="Max concurrent queries inside query_agent_benchmarking.",
    )
    args = parser.parse_args()

    missing = [v for v in ("WEAVIATE_URL", "WEAVIATE_API_KEY") if not os.getenv(v)]
    if missing:
        raise SystemExit(f"Missing required env vars: {', '.join(missing)}")

    cfg = DATASETS[args.dataset]
    extra_metrics = build_extra_metrics(args.retrieved_k, cfg)

    retriever = BaseRetriever(
        collection_name=cfg.collection,
        target_property_name=cfg.target_property,
        retrieved_k=args.retrieved_k,
        search_type="hybrid",
        verbose=False,
    )
    agent = RetrieverSearchAgent(retriever)

    out_path = (
        get_results_dir(args.dataset)
        / "extras"
        / f"hybrid_only_k{args.retrieved_k}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"dataset={args.dataset} ({cfg.qab_name})  collection={cfg.collection}\n"
        f"retrieved_k={args.retrieved_k}  extra_metrics={extra_metrics}\n"
        f"output -> {out_path}"
    )

    metrics = run_search_eval(
        search_dataset=cfg.qab_name,
        search_agent=agent,
        agent_name="hybrid_only",
        use_async=not args.sync,
        num_trials=1,
        use_subset=args.num_samples is not None,
        num_samples=args.num_samples,
        random_seed=RANDOM_SEED,
        output_path=str(out_path),
        extra_metrics=extra_metrics,
        max_concurrent=args.max_concurrent,
    )

    print(json.dumps(metrics, indent=2))
    r20 = metrics.get("recall_at_20_mean", metrics.get("recall_at_20"))
    if r20 is not None:
        print(
            f"\nRecall@20 (hybrid_only, {args.dataset}, "
            f"retrieved_k={args.retrieved_k}): {r20:.4f}"
        )


if __name__ == "__main__":
    main()
