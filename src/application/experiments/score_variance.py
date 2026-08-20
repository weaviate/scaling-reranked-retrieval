"""Measure inference-time score variance for Cohere, Voyage, and ZeroEntropy
rerankers on a few (query, top-1 hybrid doc) pairs.

Why: all three are hosted cross-encoder APIs scoring each (q, d) pair
independently. They *should* be deterministic, but in practice hosted models
can exhibit small fluctuations (batch effects, fp16 nondeterminism, server
swaps). This script quantifies that.

What it does:
    1. Loads a score cache (default: results/bright_biology/caches/k500.json)
       to pick sample queries — same queries the main experiment ranks against.
    2. For each sampled query, opens a Weaviate hybrid search and grabs the
       top-1 document's content. (Cheaper and simpler than caching content.)
    3. For each (query, doc) pair, calls each provider --trials times
       sequentially and records the raw relevance scores.
    4. Prints per-(query, provider) stats: mean, std, min, max, range.
    5. Writes raw scores + stats to
       results/bright_biology/extras/score_variance.json.

Usage:
    export WEAVIATE_URL=... WEAVIATE_API_KEY=...
    export COHERE_API_KEY=... VOYAGE_API_KEY=... ZERANK_API_KEY=...

    uv run python scripts/score_variance.py
    uv run python scripts/score_variance.py --queries 3 --trials 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
from pathlib import Path

from src.adapters.retrieval.clients import (
    get_cohere_async_client,
    get_voyage_async_client,
    get_zerank_async_client,
)
from src.adapters.retrieval.weaviate_database import async_weaviate_search_tool
from src.adapters.retrieval.providers import (
    make_async_cohere_reranker,
    make_async_voyage_reranker,
    make_async_zerank_reranker,
)
from src.adapters.retrieval.embeddings_registry import get_embedding_headers

import weaviate

from src.adapters.cache import ScoreCache
from src.config import RESULTS_DIR


COLLECTION_NAME = "BrightBiology_Default"
TARGET_PROPERTY = "content"
EMBEDDING_MODEL = "weaviate/Snowflake/snowflake-arctic-embed-l-v2.0"
_RESULTS = RESULTS_DIR / "bright_biology"
DEFAULT_CACHE = _RESULTS / "caches" / "k500.json"
DEFAULT_OUTPUT = _RESULTS / "extras" / "score_variance.json"

MODEL_OVERRIDES = {
    "cohere": "rerank-v4.0-pro",
    "voyage": "rerank-2.5",
    "zerank": "zerank-2",
}


async def fetch_top1_content(
    async_client: weaviate.WeaviateAsyncClient, query: str
) -> tuple[str, str]:
    """Return (doc_id, content) for the top hybrid result of `query`."""
    results = await async_weaviate_search_tool(
        query=query,
        collection_name=COLLECTION_NAME,
        target_property_name=TARGET_PROPERTY,
        retrieved_k=1,
        weaviate_async_client=async_client,
        return_vector=False,
        return_score=True,
    )
    if not results:
        raise RuntimeError(f"No hybrid results for query: {query[:80]!r}")
    top = results[0]
    return top.object_id, top.content


async def run_trials(
    rerank_fn, query: str, doc: str, trials: int
) -> list[float]:
    """Sequentially call rerank_fn(query, [doc], top_k=1) `trials` times."""
    scores: list[float] = []
    for _ in range(trials):
        items = await rerank_fn(query, [doc], top_k=1)
        scores.append(float(items[0].relevance_score))
    return scores


def summarize(scores: list[float]) -> dict:
    if not scores:
        return {"n": 0}
    return {
        "n": len(scores),
        "mean": statistics.fmean(scores),
        "stdev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "min": min(scores),
        "max": max(scores),
        "range": max(scores) - min(scores),
        "unique": len(set(scores)),
    }


def print_row(query: str, doc_id: str, per_provider: dict[str, dict]) -> None:
    print(f"\n--- query: {query[:90]}{'...' if len(query) > 90 else ''}")
    print(f"    top doc: {doc_id}")
    print(
        f"    {'provider':<10}{'mean':>12}{'stdev':>12}{'min':>12}{'max':>12}"
        f"{'range':>12}{'unique':>8}"
    )
    for p, stats in per_provider.items():
        print(
            f"    {p:<10}"
            f"{stats['mean']:>12.6f}"
            f"{stats['stdev']:>12.6f}"
            f"{stats['min']:>12.6f}"
            f"{stats['max']:>12.6f}"
            f"{stats['range']:>12.6f}"
            f"{stats['unique']:>8d}"
        )


async def main_async(args) -> None:
    cache = ScoreCache.load(args.cache)
    all_queries = list(cache.queries.keys())
    if not all_queries:
        raise RuntimeError(f"No queries in cache {args.cache}")

    random.seed(args.seed)
    n = min(args.queries, len(all_queries))
    sampled = random.sample(all_queries, n)

    # Weaviate (for fetching top-1 doc content)
    headers = get_embedding_headers(EMBEDDING_MODEL)
    wv = weaviate.use_async_with_weaviate_cloud(
        cluster_url=os.environ["WEAVIATE_URL"],
        auth_credentials=weaviate.auth.AuthApiKey(os.environ["WEAVIATE_API_KEY"]),
        headers=headers,
        skip_init_checks=True,
    )
    await wv.connect()

    # Reranker clients
    cohere_client = get_cohere_async_client().client
    voyage_client = get_voyage_async_client().client
    zerank_client = get_zerank_async_client().client
    rerank_fns = {
        "cohere": make_async_cohere_reranker(cohere_client, MODEL_OVERRIDES["cohere"]),
        "voyage": make_async_voyage_reranker(voyage_client, MODEL_OVERRIDES["voyage"]),
        "zerank": make_async_zerank_reranker(zerank_client, MODEL_OVERRIDES["zerank"]),
    }

    print(
        f"Sampling {n} queries x {args.trials} trials/provider "
        f"({n * args.trials} calls/provider, {n * args.trials * 3} total)"
    )

    output = {
        "metadata": {
            "cache": str(args.cache),
            "collection": COLLECTION_NAME,
            "model_overrides": MODEL_OVERRIDES,
            "n_queries": n,
            "trials_per_provider": args.trials,
            "seed": args.seed,
        },
        "results": [],
    }

    try:
        for i, query in enumerate(sampled, 1):
            print(f"\n[{i}/{n}] fetching top-1 doc...")
            doc_id, content = await fetch_top1_content(wv, query)
            if not content:
                print(f"  skipping — empty content for {doc_id}")
                continue

            # Optionally cross-check against cache's hybrid top-1 (informational).
            cache_top = cache.queries[query]["hybrid_order"][0]
            if cache_top != doc_id:
                print(
                    f"  note: cache top-1 ({cache_top}) != fresh top-1 ({doc_id}) "
                    "— hybrid isn't fully deterministic; using fresh top-1"
                )

            # Run providers in parallel; within each provider, calls are sequential
            # so we measure single-call variance rather than batched-call variance.
            cohere_scores, voyage_scores, zerank_scores = await asyncio.gather(
                run_trials(rerank_fns["cohere"], query, content, args.trials),
                run_trials(rerank_fns["voyage"], query, content, args.trials),
                run_trials(rerank_fns["zerank"], query, content, args.trials),
            )

            per_provider = {
                "cohere": summarize(cohere_scores),
                "voyage": summarize(voyage_scores),
                "zerank": summarize(zerank_scores),
            }
            print_row(query, doc_id, per_provider)

            output["results"].append({
                "query": query,
                "doc_id": doc_id,
                "doc_chars": len(content),
                "scores": {
                    "cohere": cohere_scores,
                    "voyage": voyage_scores,
                    "zerank": zerank_scores,
                },
                "stats": per_provider,
            })

    finally:
        await wv.close()
        for c in (cohere_client, voyage_client, zerank_client):
            close = getattr(c, "close", None)
            if close is None:
                continue
            try:
                r = close()
                if hasattr(r, "__await__"):
                    await r
            except Exception:
                pass

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {args.output}")

    # Aggregate across queries: average stdev per provider
    print("\n=== aggregate (mean stdev across sampled queries) ===")
    for p in ("cohere", "voyage", "zerank"):
        stdevs = [r["stats"][p]["stdev"] for r in output["results"]]
        ranges = [r["stats"][p]["range"] for r in output["results"]]
        uniques = [r["stats"][p]["unique"] for r in output["results"]]
        if stdevs:
            print(
                f"  {p:<10} mean stdev={statistics.fmean(stdevs):.6f}  "
                f"mean range={statistics.fmean(ranges):.6f}  "
                f"mean unique={statistics.fmean(uniques):.1f}/{args.trials}"
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE,
                   help="Score cache to sample queries from.")
    p.add_argument("--queries", type=int, default=5,
                   help="Number of queries to sample.")
    p.add_argument("--trials", type=int, default=10,
                   help="Inference calls per (query, provider).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
