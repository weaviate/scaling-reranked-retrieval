"""Mixture-of-rerankers experiment runner.

Sweeps the condition menu (src.domain.conditions.CONDITIONS — equal-weight
fusion only: baselines + singletons + equal-weight pair/3-way RRF+RSF fusions
across cohere/voyage/zerank; 12 conditions) against a configurable dataset
(BRIGHT subsets + IRPAPERS text). Pipeline: Weaviate hybrid search (retrieved_k) -> rerank
to reranked_k=20.

Output layout under results/{results_subdir}/
(results_subdir is per-dataset: e.g. bright_biology, irpapers_text):
    caches/k{N}.json                       -- score cache from --collect-only
    runs/k{N}_from_k{M}.json               -- derived eval at retrieved_k=N
                                              from a cache collected at M
    extras/collect_k{N}.json               -- side-effect metrics from collect
    extras/...                             -- one-off artifacts

Usage:
    uv run python scripts/run_experiment.py --dataset biology
    uv run python scripts/run_experiment.py --dataset earth_science --collect-only --retrieved-k 2000
    uv run python scripts/run_experiment.py --dataset earth_science --from-cache .../caches/k2000.json --retrieved-k 500
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from query_agent_benchmarking import run_search_eval

from src.adapters import qab

# qab>=0.7 guard + per-dataset loader memoization (30 run_search_eval calls
# per sweep would otherwise reload the corpus each time).
qab.setup()

from src.adapters.retrieval.clients import (  # noqa: E402
    get_cohere_async_client,
    get_cohere_client,
    get_voyage_async_client,
    get_voyage_client,
    get_zerank_async_client,
    get_zerank_client,
)
from src.adapters.retrieval.base_retriever import BaseRetriever  # noqa: E402
from src.adapters.retrieval.providers import (  # noqa: E402
    configure_voyage_post_call_sleep,
)
from src.adapters.retrieval.cross_encoder_reranker import (  # noqa: E402
    CrossEncoderReranker,
)

from src.adapters.qab import RetrieverSearchAgent  # noqa: E402
from src.application.collect import CollectScoresAgent  # noqa: E402
from src.application.derived import DerivedSearchAgent  # noqa: E402
from src.adapters.cache import ScoreCache, validate_cache_for_use  # noqa: E402
from src.domain.conditions import CONDITIONS, Condition  # noqa: E402
from src.config import (  # noqa: E402
    DATASETS,
    DEFAULT_RERANKED_K,
    DEFAULT_RETRIEVED_K,
    MODEL_OVERRIDES,
    RANDOM_SEED,
    get_results_dir,
)
from src.domain.metrics import build_extra_metrics  # noqa: E402


def build_retriever(
    condition: Condition,
    retrieved_k: int,
    reranked_k: int,
    collection_name: str,
    target_property: str,
):
    """Build the retriever for one condition.

    Returns a plain BaseRetriever (hybrid search, no rerank) when
    condition.provider is None, otherwise a CrossEncoderReranker.
    hybrid_only's retrieved_k tracks the experiment's retrieved_k so that
    Recall@{50,100,200,500} measure the actual first-stage ceiling.
    """
    if condition.provider is None:
        return BaseRetriever(
            collection_name=collection_name,
            target_property_name=target_property,
            retrieved_k=retrieved_k,
            search_type="hybrid",
            verbose=False,
        )

    # Determine which rerankers this condition needs.
    if condition.provider == "hybrid":
        needed: tuple[str, ...] = condition.rerankers
    elif condition.provider in ("cohere", "voyage", "zerank"):
        needed = (condition.provider,)
    else:
        needed = ()

    reranker_clients = []
    if "cohere" in needed:
        reranker_clients.extend([get_cohere_client(), get_cohere_async_client()])
    if "voyage" in needed:
        reranker_clients.extend([get_voyage_client(), get_voyage_async_client()])
    if "zerank" in needed:
        reranker_clients.extend([get_zerank_client(), get_zerank_async_client()])

    return CrossEncoderReranker(
        collection_name=collection_name,
        target_property_name=target_property,
        reranker_clients=reranker_clients,
        retrieved_k=retrieved_k,
        reranked_k=reranked_k,
        provider=condition.provider,
        hybrid_weights=condition.weights,
        fusion_method=condition.fusion_method or "rrf",
        model_name_overrides=MODEL_OVERRIDES,
        verbose=False,
    )


def run_condition(
    condition: Condition,
    agent,
    num_samples: Optional[int],
    use_async: bool,
    retrieved_k: int,
    dataset_name: str,
    scratch_path: Path,
    extra_metrics: list[dict],
    max_concurrent: int,
) -> dict:
    """Run a single condition through query_agent_benchmarking.

    Per-condition metrics are written by run_search_eval to scratch_path; the
    aggregated summary is built by the caller via write_summary. scratch files
    are kept for crash-recovery / spot-debugging but live under runs/.scratch/.
    """
    print(f"\n=== {condition.name} (retrieved_k={retrieved_k}) ===")
    if condition.provider is None:
        print("  provider=none (hybrid search, no rerank)")
    else:
        print(
            f"  provider={condition.provider} "
            f"fusion={condition.fusion_method} weights={condition.weights}"
        )
    print(f"  agent={type(agent).__name__}")

    scratch_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = run_search_eval(
        search_dataset=dataset_name,
        search_agent=agent,
        agent_name=condition.name,
        use_async=use_async,
        num_trials=1,
        use_subset=num_samples is not None,
        num_samples=num_samples,
        random_seed=RANDOM_SEED,
        output_path=str(scratch_path),
        extra_metrics=extra_metrics,
        max_concurrent=max_concurrent,
    )

    print(f"  metrics saved -> {scratch_path}")
    return metrics


def write_summary(
    results: dict[str, dict],
    retrieved_k: int,
    reranked_k: int,
    dataset_name: str,
    collection_name: str,
    summary_path: Path,
    cache_retrieved_k: Optional[int] = None,
) -> None:
    """Write a single summary JSON containing all condition metrics.

    cache_retrieved_k is recorded when this is a derived run (from --from-cache);
    None for real-call or collect-only summaries.
    """
    payload = {
        "dataset": dataset_name,
        "collection": collection_name,
        "retrieved_k": retrieved_k,
        "reranked_k": reranked_k,
        "random_seed": RANDOM_SEED,
        "model_overrides": MODEL_OVERRIDES,
        "cache_retrieved_k": cache_retrieved_k,
        "results": results,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSummary written to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS.keys()),
        default="biology",
        help=(
            "Dataset slug. Selects the DatasetConfig (qab dataset, Weaviate "
            "collection, target property, results dir) from DATASETS. Results "
            "are written under results/{results_subdir}/. Default: biology."
        ),
    )
    parser.add_argument(
        "--retrieved-k",
        type=int,
        default=DEFAULT_RETRIEVED_K,
        help=(
            "First-stage candidate pool size. Hybrid_only returns this many "
            "docs (so Recall@retrieved_k is measurable). Reranked conditions "
            "rerank this pool down to reranked_k. Default: 100."
        ),
    )
    parser.add_argument(
        "--reranked-k",
        type=int,
        default=DEFAULT_RERANKED_K,
        help=(
            "Post-rerank output cap. Reranked conditions return this many "
            "docs, so Recall@K is measurable for K ≤ reranked_k (caps at "
            "Recall@reranked_k for K > reranked_k). Default 20. Use 100 to "
            "enable Recall@50 / Recall@100 measurement; output is routed to "
            "results/.../runs_rk{N}/ so the default reranked_k=20 sweeps "
            "stay addressable. Because cross-encoder scoring is per-(query, "
            "doc) independent, running at reranked_k=100 produces R@1/R@5/"
            "R@20 numbers byte-identical to a reranked_k=20 run on the same "
            "cache — no need to re-derive shallower cutoffs."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a 10-sample smoke test of the first condition only.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated condition names to run (default: all).",
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
        help=(
            "Max concurrent queries inside query_agent_benchmarking. "
            "Default 1 (serialized) to keep Voyage TPM under control at "
            "large retrieved_k. Override only if you've verified your rate "
            "limits allow it."
        ),
    )
    parser.add_argument(
        "--voyage-sleep-seconds",
        type=float,
        default=30.0,
        help=(
            "Sleep this long after every successful Voyage rerank operation "
            "to stay under the 4M-tokens-per-minute cap. At retrieved_k=2000 "
            "each query bursts ~1M Voyage tokens, so 30s yields ~1.8M TPM "
            "(comfortable margin). Set to 0 to disable. Default 30."
        ),
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help=(
            "Score-collection mode. Runs Weaviate + Cohere + Voyage + Zerank "
            "once at retrieved_k and writes "
            "results/{results_subdir}/caches/k{N}.json. Skips the per-condition "
            "eval — the cache is the artifact. Use --from-cache afterwards to "
            "derive results at any retrieved_k <= N."
        ),
    )
    parser.add_argument(
        "--from-cache",
        type=str,
        default=None,
        help=(
            "Path to a score cache JSON produced by --collect-only. Derives "
            "per-condition rankings locally without any API calls. The cache's "
            "retrieved_k must be >= --retrieved-k, and model overrides / "
            "dataset / collection must match."
        ),
    )
    args = parser.parse_args()

    required = ["WEAVIATE_URL", "WEAVIATE_API_KEY"]
    if not args.from_cache:
        # Real or collect modes both need reranker keys
        required += ["COHERE_API_KEY", "VOYAGE_API_KEY", "ZERANK_API_KEY"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        raise SystemExit(f"Missing required env vars: {', '.join(missing)}")

    retrieved_k = args.retrieved_k
    reranked_k = args.reranked_k
    dataset_slug = args.dataset
    cfg = DATASETS[dataset_slug]
    dataset_name, collection_name = cfg.qab_name, cfg.collection
    target_property = cfg.target_property
    results_dir = get_results_dir(dataset_slug)
    extra_metrics = build_extra_metrics(retrieved_k, cfg)
    configure_voyage_post_call_sleep(args.voyage_sleep_seconds)
    # Default reranked_k=20 writes to runs/; any other value writes to a sibling
    # runs_rk{N}/ so prior reranked_k=20 sweeps stay addressable.
    runs_subdir = "runs" if reranked_k == DEFAULT_RERANKED_K else f"runs_rk{reranked_k}"
    print(
        f"dataset={dataset_slug} ({dataset_name})  collection={collection_name}"
    )
    print(
        f"retrieved_k={retrieved_k}  reranked_k={reranked_k}  "
        f"results_dir={results_dir}  runs_subdir={runs_subdir}  "
        f"extra_metrics={extra_metrics}"
    )
    print(
        f"max_concurrent={args.max_concurrent}  "
        f"voyage_sleep={args.voyage_sleep_seconds}s"
    )

    # ----------------- Collect-only mode ----------------- #
    if args.collect_only:
        if args.from_cache:
            raise SystemExit("--collect-only and --from-cache are mutually exclusive.")

        cache_path = results_dir / "caches" / f"k{retrieved_k}.json"
        existing = ScoreCache.load(cache_path) if cache_path.exists() else None
        cache = existing or ScoreCache(
            metadata={
                "dataset": dataset_name,
                "collection": collection_name,
                "retrieved_k": retrieved_k,
                "model_overrides": MODEL_OVERRIDES,
            },
            queries={},
        )
        if existing:
            print(
                f"Resuming collection: {len(cache.queries)} queries already cached "
                f"at {cache_path}."
            )

        collector = CollectScoresAgent(
            collection_name=collection_name,
            target_property=target_property,
            retrieved_k=retrieved_k,
            cache=cache,
            cache_path=cache_path,
            cohere_model=MODEL_OVERRIDES["cohere"],
            voyage_model=MODEL_OVERRIDES["voyage"],
            zerank_model=MODEL_OVERRIDES["zerank"],
        )

        out_path = results_dir / "extras" / f"collect_k{retrieved_k}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        run_search_eval(
            search_dataset=dataset_name,
            search_agent=collector,
            agent_name="collect_scores",
            use_async=not args.sync,
            num_trials=1,
            use_subset=args.num_samples is not None,
            num_samples=args.num_samples,
            random_seed=RANDOM_SEED,
            output_path=str(out_path),
            max_concurrent=args.max_concurrent,
        )
        print(
            f"\nCollection done. {len(cache.queries)} queries cached -> {cache_path}"
        )
        return

    # ----------------- From-cache (derived) mode ----------------- #
    cache: Optional[ScoreCache] = None
    cache_retrieved_k: Optional[int] = None
    if args.from_cache:
        cache_path = Path(args.from_cache)
        cache = ScoreCache.load(cache_path)
        validate_cache_for_use(
            cache,
            needed_retrieved_k=retrieved_k,
            expected_model_overrides=MODEL_OVERRIDES,
            expected_dataset=dataset_name,
            expected_collection=collection_name,
        )
        cache_retrieved_k = cache.metadata.get("retrieved_k")
        print(
            f"Loaded cache: {len(cache.queries)} queries, "
            f"retrieved_k={cache_retrieved_k}"
        )

    # ----------------- Condition selection (shared) ----------------- #
    conditions = CONDITIONS
    if args.smoke:
        conditions = CONDITIONS[:1]
        args.num_samples = args.num_samples or 10
    elif args.only:
        wanted = {n.strip() for n in args.only.split(",")}
        conditions = [c for c in CONDITIONS if c.name in wanted]
        if not conditions:
            raise SystemExit(f"No matching conditions in --only={args.only}")

    # Naming for the summary file + per-condition scratch dir. Derived runs
    # encode provenance ("from_k{M}"); real-call runs are tagged "_real".
    if args.from_cache:
        run_label = f"k{retrieved_k}_from_k{cache_retrieved_k}"
    else:
        run_label = f"k{retrieved_k}_real"
    summary_path = results_dir / runs_subdir / f"{run_label}.json"
    scratch_dir = results_dir / runs_subdir / ".scratch" / run_label
    print(f"Run label: {run_label}  summary -> {summary_path}")

    results: dict[str, dict] = {}
    for cond in conditions:
        if args.from_cache:
            assert cache is not None  # narrowed by args.from_cache check
            agent = DerivedSearchAgent(
                cache=cache,
                retrieved_k=retrieved_k,
                condition=cond,
                reranked_k=reranked_k,
            )
        else:
            retriever = build_retriever(
                cond, retrieved_k, reranked_k, collection_name, target_property
            )
            agent = RetrieverSearchAgent(retriever)

        try:
            results[cond.name] = run_condition(
                cond,
                agent=agent,
                num_samples=args.num_samples,
                use_async=not args.sync,
                retrieved_k=retrieved_k,
                dataset_name=dataset_name,
                scratch_path=scratch_dir / f"{cond.name}.json",
                extra_metrics=extra_metrics,
                max_concurrent=args.max_concurrent,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {e!r}")
            results[cond.name] = {"error": repr(e)}

    write_summary(
        results,
        retrieved_k=retrieved_k,
        reranked_k=reranked_k,
        dataset_name=dataset_name,
        collection_name=collection_name,
        summary_path=summary_path,
        cache_retrieved_k=cache_retrieved_k,
    )


if __name__ == "__main__":
    main()
