"""Latency + payload measurement harness (MoCE §3.3 / §3.4).

Standalone, timing-only companion to run_experiment.py. Produces the two
numbers the paper stubs:

  §3.4 — rerank latency vs. K. Per-provider wall-clock latency to score a
         candidate pool, swept over K ∈ {100,200,500,1000,2000}, so the
         test-time-compute framing can state latency as a function of pool
         depth and back the "added latency is bounded by the slowest model,
         not the ensemble size" claim (parallel_bound vs serial_sum).

  §3.3 — hybrid query latency + payload at k=2000. Single-query Weaviate
         hybrid search wall-clock and two byte figures (the serialized wire
         payload of the retrieval call, and the raw retrieved-content size the
         rerankers ingest), at retrieved_k=2000.

Both measurements run over the SAME randomly sampled queries per dataset, so
the query set and sampling logic are written once.

This script is TIMING ONLY: it never reads, writes, or invalidates any
caches/k{N}.json score cache (a startup assertion enforces that the output
path is not under caches/). Output lands under results/latency/ exclusively.

Reuses (does not reimplement):
  - DATASETS / DatasetConfig + get_results_dir from src.config
  - the all-three-present intersection (build_query_set) the agreement / oracle
    scripts use, so every sampled query has a valid timing on all three
    providers
  - the reranker client constructors from clients.py and the per-provider
    chunking constants from chunking.py/providers.py (NOT the reactive
    byte/token halving — see _rerank_zerank / the failure handling below)
  - the same query_agent_benchmarking>=0.7 guard (src.adapters.qab)

Usage:
    uv run python scripts/latency_measurement.py --dataset robotics
    uv run python scripts/latency_measurement.py --all-datasets
    # flags: --n-queries 5 --seed 42 --k-sweep 100,200,500,1000,2000
    #        --repeats 3 --smoke --network-location "us-east laptop"

Env (all five required — this is a live-call script):
    WEAVIATE_URL, WEAVIATE_API_KEY, COHERE_API_KEY, VOYAGE_API_KEY, ZERANK_API_KEY
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import math
import os
import random
import statistics
import time
from pathlib import Path
from typing import Callable, Optional

# Reuse the dataset registry + results helpers (single source of truth) and the
# exact intersection-building logic the agreement/oracle scripts use.
from src.adapters.cache import ScoreCache, validate_cache_for_use
from src.config import (
    CACHE_K,
    DATASETS,
    EMBEDDING_MODEL,
    MODEL_OVERRIDES,
    PROVIDERS,
    RESULTS_DIR,
)
from src.application.queryset import build_query_set

from src.adapters.retrieval.clients import (
    get_cohere_async_client,
    get_voyage_async_client,
    get_zerank_async_client,
)
from src.adapters.retrieval.weaviate_database import (
    async_weaviate_search_tool,
)
from src.adapters.retrieval.embeddings_registry import (
    get_embedding_headers,
)

# Per-provider per-call doc-count limit and zerank's byte budget, reused so the
# chunking matches the experiment. We deliberately use the PLAIN call path (the
# proactive chunking only) and NOT the safe_call_chunk reactive halving: a
# reactive halving retry on a byte/token overflow would replay the failed
# round-trip and double-count latency, so on overflow we record the cell as a
# null failure instead (see the per-(query,K,provider) loop below).
from src.adapters.retrieval.chunking import (
    MAX_DOCS_PER_CALL as _MAX_DOCS_PER_CALL,
    byte_budget_chunks as _byte_budget_chunks,
)
from src.adapters.retrieval.providers import (
    ZERANK_BYTE_BUDGET as _ZERANK_BYTE_BUDGET,
)

from src.adapters import qab

# qab>=0.7 guard + load_search_dataset memoization, so the per-dataset qab
# query load happens at most once per process.
qab.setup()

DEFAULT_K_SWEEP = (100, 200, 500, 1000, 2000)
DEFAULT_N_QUERIES = 5
DEFAULT_SEED = 42
DEFAULT_REPEATS = 3
HYBRID_K = 2000  # measurement-2 pool size (and the largest slice for meas. 1)

# Fixed sleep between successive timed API calls, to keep provider-side
# rate-limit throttling from skewing a timing. Applied after every rerank call
# and between hybrid-search reps.
INTER_CALL_SLEEP_S = 0.2


# --------------------------------------------------------------------------- #
# Small stats helpers (no numpy dependency)                                    #
# --------------------------------------------------------------------------- #


def _percentile(vals: list[float], p: float) -> Optional[float]:
    """Linear-interpolation percentile, p in [0,1]. None on empty input."""
    if not vals:
        return None
    s = sorted(vals)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(s[int(k)])
    return float(s[f] * (c - k) + s[c] * (k - f))


def _per_doc_stats(vals: list[float]) -> dict:
    """mean/median/p95/max over the pooled single-doc byte sizes."""
    if not vals:
        return {"mean": None, "median": None, "p95": None, "max": None,
                "n_docs_pooled": 0}
    p95 = _percentile(vals, 0.95)
    return {
        "mean": round(statistics.mean(vals), 1),
        "median": round(statistics.median(vals), 1),
        "p95": round(p95, 1) if p95 is not None else None,
        "max": max(vals),
        "n_docs_pooled": len(vals),
    }


def _summ(vals: list[float], ndigits: int = 3) -> dict:
    """median/min/max over vals (rounded), or all-null on empty."""
    if not vals:
        return {"median": None, "min": None, "max": None}
    return {
        "median": round(statistics.median(vals), ndigits),
        "min": round(min(vals), ndigits),
        "max": round(max(vals), ndigits),
    }


# --------------------------------------------------------------------------- #
# Per-provider timed rerank (plain chunking, no reactive halving)              #
# --------------------------------------------------------------------------- #
#
# Each function issues the provider's rerank API call(s) and returns when the
# response(s) are fully received and parsed (awaiting the SDK call yields the
# parsed response object). When the pool exceeds the per-call limit the chunks
# are run concurrently with asyncio.gather, mirroring the experiment's fan-out
# — so the measured latency for a 2000-doc pool is ~the latency of one
# 1000-doc call, the real deployment cost, not the serial sum of the chunks.


async def _rerank_cohere(client, model: str, query: str, docs: list[str]) -> None:
    limit = _MAX_DOCS_PER_CALL["cohere"]
    chunks = [docs[i : i + limit] for i in range(0, len(docs), limit)]
    await asyncio.gather(
        *(
            client.rerank(model=model, query=query, documents=c, top_n=len(c))
            for c in chunks
        )
    )


async def _rerank_voyage(client, model: str, query: str, docs: list[str]) -> None:
    limit = _MAX_DOCS_PER_CALL["voyage"]
    chunks = [docs[i : i + limit] for i in range(0, len(docs), limit)]
    await asyncio.gather(
        *(
            client.rerank(query=query, documents=c, model=model, top_k=len(c))
            for c in chunks
        )
    )


async def _rerank_zerank(client, model: str, query: str, docs: list[str]) -> None:
    # Byte-budget chunking (the proactive zerank packing), NO reactive halving.
    chunks = _byte_budget_chunks(docs, _MAX_DOCS_PER_CALL["zerank"], _ZERANK_BYTE_BUDGET)
    await asyncio.gather(
        *(
            client.models.rerank(model=model, query=query, documents=c, top_n=len(c))
            for _, c in chunks
        )
    )


RERANK_FNS: dict[str, Callable] = {
    "cohere": _rerank_cohere,
    "voyage": _rerank_voyage,
    "zerank": _rerank_zerank,
}


# --------------------------------------------------------------------------- #
# Cell construction                                                            #
# --------------------------------------------------------------------------- #


def _per_query_entry(qid: str, samples_ms: list[float], n_fail: int,
                     error: Optional[str]) -> dict:
    """One (query, K, provider) result: median-of-N-reps with min/max retained."""
    s = _summ(samples_ms)
    return {
        "query_id": qid,
        "samples_ms": [round(x, 3) for x in samples_ms],
        "median": s["median"],
        "min": s["min"],
        "max": s["max"],
        "n_fail": n_fail,
        "error": error,
    }


def _cell_from_per_query(per_query: list[dict]) -> dict:
    """Roll a (provider, K) cell up from its per-query median-of-N values.

    median/min/max are taken across the per-query medians; n_ok / n_fail count
    queries (a query is ok if at least one of its reps succeeded). The full
    per_query list (each a median of `repeats` samples with its own min/max) is
    retained so the per-(query,K,provider) detail is auditable.
    """
    medians = [pq["median"] for pq in per_query if pq["median"] is not None]
    n_ok = len(medians)
    n_fail = len(per_query) - n_ok
    summ = _summ(medians)
    return {
        "median": summ["median"],
        "min": summ["min"],
        "max": summ["max"],
        "n_ok": n_ok,
        "n_fail": n_fail,
        "per_query": per_query,
    }


# --------------------------------------------------------------------------- #
# Provider client lifecycle                                                    #
# --------------------------------------------------------------------------- #


class Providers:
    """Holds the three async reranker clients + their experiment model ids."""

    def __init__(self) -> None:
        self.clients: dict = {}
        self.models = {p: MODEL_OVERRIDES[p] for p in PROVIDERS}

    async def open(self) -> None:
        self.clients["cohere"] = get_cohere_async_client().client
        self.clients["voyage"] = get_voyage_async_client().client
        self.clients["zerank"] = get_zerank_async_client().client

    async def close(self) -> None:
        for client in self.clients.values():
            close = getattr(client, "close", None)
            if close is None:
                continue
            try:
                result = close()
                if hasattr(result, "__await__"):
                    await result
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# Per-dataset measurement                                                      #
# --------------------------------------------------------------------------- #


def _resolve_sampled_queries(
    slug: str, n_queries: int, seed: int
) -> list[dict]:
    """Sample queries from the all-three-present intersection (cache required).

    Returns [{text, qid}, ...]. Raises if the k=2000 cache is missing.
    """
    cfg = DATASETS[slug]
    cache_path = RESULTS_DIR / cfg.results_subdir / "caches" / f"k{CACHE_K}.json"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"No k{CACHE_K} cache at {cache_path}. The intersection (all-three-"
            "present) query set is derived from it; collect it first via "
            f"run_experiment.py --dataset {slug} --retrieved-k {CACHE_K} --collect-only."
        )
    cache = ScoreCache.load(cache_path)
    validate_cache_for_use(
        cache,
        needed_retrieved_k=CACHE_K,
        expected_model_overrides=MODEL_OVERRIDES,
        expected_dataset=cfg.qab_name,
        expected_collection=cfg.collection,
    )
    qs = build_query_set(cache, cfg.qab_name)
    # Sort the intersection texts before sampling so the draw is reproducible
    # regardless of cache insertion order.
    intersection = sorted(qs.gold.keys())
    if not intersection:
        raise RuntimeError(f"Empty all-three-present intersection for {slug}.")
    n = min(n_queries, len(intersection))
    if n < n_queries:
        print(f"  [warn] requested {n_queries} queries but intersection has "
              f"only {len(intersection)}; sampling {n}.")
    rng = random.Random(seed)
    texts = rng.sample(intersection, n)
    return [{"text": t, "qid": qs.query_ids.get(t, t[:64])} for t in texts]


async def _connect_weaviate(headers: dict):
    import weaviate

    client = weaviate.use_async_with_weaviate_cloud(
        cluster_url=os.environ["WEAVIATE_URL"],
        auth_credentials=weaviate.auth.AuthApiKey(os.environ["WEAVIATE_API_KEY"]),
        headers=headers,
        skip_init_checks=True,
    )
    await client.connect()
    return client


async def measure_dataset(
    slug: str,
    providers: Providers,
    weaviate_client,
    *,
    n_queries: int,
    seed: int,
    k_sweep: tuple[int, ...],
    repeats: int,
    hybrid_alpha: Optional[float],
    network_location: str,
    priced_date: str,
) -> dict:
    cfg = DATASETS[slug]
    target_property = cfg.target_property
    sampled = _resolve_sampled_queries(slug, n_queries, seed)
    sampled_query_ids = [q["qid"] for q in sampled]
    print(f"\n=== {slug} ({cfg.qab_name}) collection={cfg.collection} ===")
    print(f"  sampled {len(sampled)} queries (seed={seed}): {sampled_query_ids}")
    print(f"  k_sweep={list(k_sweep)} repeats={repeats}")

    # results[provider][K] -> list of per-query entries.
    results: dict[str, dict[int, list[dict]]] = {
        p: {K: [] for K in k_sweep} for p in PROVIDERS
    }
    # Measurement 2 accumulators.
    hybrid_latency_per_query: list[float] = []      # median-of-reps per query
    response_payload_per_query: list[float] = []
    content_total_per_query: list[float] = []
    query_bytes_per_query: list[float] = []
    per_doc_bytes_pool: list[float] = []            # pooled single-doc byte sizes
    n_docs_per_query: list[int] = []

    for qi, q in enumerate(sampled):
        text = q["text"]
        qid = q["qid"]

        # ---- Measurement 2: time the k=2000 hybrid search (repeats×), keep the
        # last retrieval's docs as the shared candidate pool for measurement 1.
        ret_samples: list[float] = []
        sources = None
        for rep in range(repeats):
            t0 = time.perf_counter()
            sources = await async_weaviate_search_tool(
                query=text,
                collection_name=cfg.collection,
                target_property_name=target_property,
                weaviate_async_client=weaviate_client,
                retrieved_k=HYBRID_K,
                return_score=True,
                return_vector=False,
                search_type="hybrid",
                hybrid_alpha=hybrid_alpha,
            )
            ret_samples.append((time.perf_counter() - t0) * 1000.0)
            await asyncio.sleep(INTER_CALL_SLEEP_S)

        assert sources is not None  # repeats >= 1, so the loop ran at least once
        pool_texts = [s.content for s in sources]
        n_docs = len(pool_texts)
        n_docs_per_query.append(n_docs)

        # Wire payload of the retrieval call: serialize the fields actually
        # fetched (doc id + the requested content property + score) per hit.
        payload_repr = [
            {"object_id": s.object_id, target_property: s.content,
             "score": s.relevance_score}
            for s in sources
        ]
        response_payload_bytes = len(
            json.dumps(payload_repr, ensure_ascii=False).encode("utf-8")
        )
        # Raw retrieved-content bytes: the document payload downstream stages
        # (incl. the three rerankers) must move — query-shape-independent.
        doc_bytes = [len(s.content.encode("utf-8")) for s in sources]
        content_total = sum(doc_bytes)
        query_bytes = len(text.encode("utf-8"))

        hybrid_latency_per_query.append(round(statistics.median(ret_samples), 3))
        response_payload_per_query.append(response_payload_bytes)
        content_total_per_query.append(content_total)
        query_bytes_per_query.append(query_bytes)
        per_doc_bytes_pool.extend(doc_bytes)

        print(f"  [{qi + 1}/{len(sampled)}] {qid}: hybrid k{HYBRID_K} "
              f"median={statistics.median(ret_samples):.1f}ms, {n_docs} docs, "
              f"content={content_total / 1e6:.2f}MB")

        # ---- Measurement 1: rerank latency vs K. Slice the shared pool to each
        # K (docs identical across the three providers within a (query,K)), then
        # time each provider's rerank sequentially with the inter-call sleep.
        for K in k_sweep:
            docs = pool_texts[:K]
            k_report: list[str] = []
            for provider in PROVIDERS:
                samples: list[float] = []
                n_fail = 0
                error: Optional[str] = None
                fn = RERANK_FNS[provider]
                client = providers.clients[provider]
                model = providers.models[provider]
                for rep in range(repeats):
                    t0 = time.perf_counter()
                    try:
                        await fn(client, model, text, docs)
                        samples.append((time.perf_counter() - t0) * 1000.0)
                    except Exception as e:  # noqa: BLE001
                        # null-with-reason (content filter / byte- or token-limit
                        # overflow / rate-limit). Do NOT reactively halve+retry —
                        # that would double-count latency.
                        n_fail += 1
                        error = f"{type(e).__name__}: {str(e)[:200]}"
                    await asyncio.sleep(INTER_CALL_SLEEP_S)
                entry = _per_query_entry(qid, samples, n_fail, error)
                results[provider][K].append(entry)
                med = entry["median"]
                k_report.append(
                    f"{provider}={med:.0f}ms" if med is not None
                    else f"{provider}=FAIL"
                )
            print(f"      K={K}: " + "  ".join(k_report), flush=True)

    # ---- Build rerank_latency_ms + ensemble_latency_ms.
    rerank_latency_ms: dict[str, dict[str, dict]] = {p: {} for p in PROVIDERS}
    for provider in PROVIDERS:
        for K in k_sweep:
            rerank_latency_ms[provider][str(K)] = _cell_from_per_query(
                results[provider][K]
            )

    ensemble_latency_ms: dict[str, dict] = {}
    for K in k_sweep:
        medians = [
            rerank_latency_ms[p][str(K)]["median"]
            for p in PROVIDERS
            if rerank_latency_ms[p][str(K)]["median"] is not None
        ]
        ensemble_latency_ms[str(K)] = {
            "parallel_bound": round(max(medians), 3) if medians else None,
            "serial_sum": round(sum(medians), 3) if medians else None,
            "providers_ok": len(medians),
        }

    # ---- Measurement 2 rollup.
    hybrid_k2000 = {
        "latency_ms": {
            **_summ(hybrid_latency_per_query),
            "per_query": hybrid_latency_per_query,
        },
        "response_payload_bytes": {
            **_summ(response_payload_per_query, ndigits=0),
            "per_query": response_payload_per_query,
            "properties_fetched": [target_property],
        },
        "retrieved_content_bytes": {
            "total": {
                **_summ(content_total_per_query, ndigits=0),
                "per_query": content_total_per_query,
            },
            "per_doc": _per_doc_stats(per_doc_bytes_pool),
        },
        "query_bytes": {
            **_summ(query_bytes_per_query, ndigits=0),
            "per_query": query_bytes_per_query,
        },
        "n_docs_per_query": n_docs_per_query,
    }

    return {
        "dataset": slug,
        "qab_name": cfg.qab_name,
        "collection": cfg.collection,
        "target_property": target_property,
        "seed": seed,
        "n_queries": len(sampled),
        "k_sweep": list(k_sweep),
        "repeats": repeats,
        "sampled_query_ids": sampled_query_ids,
        "network_location": network_location,
        "priced_date": priced_date,
        "model_overrides": dict(MODEL_OVERRIDES),
        "search_type": "hybrid",
        "hybrid_alpha": hybrid_alpha,  # None => Weaviate hybrid default (matches experiment)
        "rerank_latency_ms": rerank_latency_ms,
        "ensemble_latency_ms": ensemble_latency_ms,
        "hybrid_k2000": hybrid_k2000,
    }


# --------------------------------------------------------------------------- #
# Output                                                                       #
# --------------------------------------------------------------------------- #


def _latency_dir() -> Path:
    out = RESULTS_DIR / "latency"
    # Acceptance guard: this script is timing-only and must never touch the
    # score caches. Refuse if the output path is anywhere under a caches/ dir.
    assert "caches" not in out.parts, "latency output must not live under caches/"
    return out


def write_dataset_json(payload: dict) -> Path:
    out_dir = _latency_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"latency_{payload['dataset']}.json"
    assert "caches" not in path.parts
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  wrote {path}")
    return path


def _fmt_ms(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.0f}"


def write_latency_md(payloads: dict[str, dict]) -> Path:
    """Cross-dataset LATENCY.md: rerank latency vs K + hybrid k=2000 payload."""
    datasets = [d for d in DATASETS if d in payloads]
    any_payload = payloads[datasets[0]]
    k_sweep = any_payload["k_sweep"]
    lines: list[str] = []
    A = lines.append

    A("# Latency + Payload Measurements (MoCE §3.3 / §3.4)")
    A("")
    A("Per-provider rerank latency vs. candidate-pool depth (§3.4) and "
      "single-query hybrid-search latency + payload at k=2000 (§3.3), over "
      "the same randomly sampled queries per dataset. All numbers are "
      "end-to-end API latency from the experiment's network location "
      "(**not** compute-only). See each `latency_{dataset}.json` for "
      "per-query detail, failure reasons, and reproducibility fields.")
    A("")
    A(f"- Network location: `{any_payload['network_location']}`")
    A(f"- Priced date: `{any_payload['priced_date']}`")
    A(f"- Seed: {any_payload['seed']}  ·  queries/dataset: "
      f"{any_payload['n_queries']}  ·  repeats: {any_payload['repeats']} "
      "(median reported, min/max retained in JSON)")
    models_str = ", ".join(
        f"{p}={any_payload['model_overrides'][p]}" for p in PROVIDERS
    )
    A(f"- Models: {models_str}")
    A("")

    # Table 1: rerank latency vs K per provider (median ms across queries).
    A("## §3.4 — Rerank latency vs. K (median ms)")
    A("")
    A("Per (dataset, provider): median across the sampled queries of the "
      "per-query median-of-repeats rerank latency, at each K. The "
      "`ensemble (parallel_bound)` row is the max over the three providers — "
      "the deployment latency, since the three passes fan out in parallel; "
      "`ensemble (serial_sum)` is the naive sequential cost. The gap between "
      "them is what parallelism saves.")
    A("")
    header = "| Dataset | Provider | " + " | ".join(f"K={k}" for k in k_sweep) + " |"
    A(header)
    A("|---|---|" + "|".join(["---"] * len(k_sweep)) + "|")
    for d in datasets:
        p = payloads[d]
        for provider in PROVIDERS:
            cells = " | ".join(
                _fmt_ms(p["rerank_latency_ms"][provider][str(k)]["median"])
                for k in k_sweep
            )
            A(f"| {d} | {provider} | {cells} |")
        pb = " | ".join(
            _fmt_ms(p["ensemble_latency_ms"][str(k)]["parallel_bound"]) for k in k_sweep
        )
        ss = " | ".join(
            _fmt_ms(p["ensemble_latency_ms"][str(k)]["serial_sum"]) for k in k_sweep
        )
        A(f"| {d} | ensemble (parallel_bound) | {pb} |")
        A(f"| {d} | ensemble (serial_sum) | {ss} |")
    A("")

    # Table 2: hybrid k=2000 latency + payload per dataset.
    A("## §3.3 — Hybrid k=2000 latency + payload (median across queries)")
    A("")
    A("`hybrid latency` is median single-query wall-clock for a k=2000 hybrid "
      "search. `response payload` is the serialized wire size of that "
      "retrieval response (ids + `content` + scores). `content total` is the "
      "raw UTF-8 byte size of the retrieved document content summed over the "
      "pool (the bytes the rerankers ingest); `per-doc` stats pool every "
      "retrieved document across the sampled queries. Effective per-reranker "
      "transfer ≈ `content total + query_bytes × n_docs` (ZeroEntropy counts "
      "the query bytes once per document).")
    A("")
    A("| Dataset | hybrid latency (ms) | response payload (MB) | content total (MB) "
      "| per-doc mean (B) | per-doc p95 (B) | per-doc max (B) | query (B) |")
    A("|---|---|---|---|---|---|---|---|")
    for d in datasets:
        h = payloads[d]["hybrid_k2000"]
        lat = h["latency_ms"]["median"]
        rp = h["response_payload_bytes"]["median"]
        ct = h["retrieved_content_bytes"]["total"]["median"]
        pd = h["retrieved_content_bytes"]["per_doc"]
        qb = h["query_bytes"]["median"]
        A(
            f"| {d} | {_fmt_ms(lat)} | "
            f"{(rp / 1e6):.2f} | {(ct / 1e6):.2f} | "
            f"{_fmt_ms(pd['mean'])} | {_fmt_ms(pd['p95'])} | {_fmt_ms(pd['max'])} | "
            f"{_fmt_ms(qb)} |"
        )
    A("")

    out_dir = _latency_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "LATENCY.md"
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}")
    return path


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def _parse_k_sweep(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x.strip())


async def _run(args) -> None:
    for var in ("WEAVIATE_URL", "WEAVIATE_API_KEY", "COHERE_API_KEY",
                "VOYAGE_API_KEY", "ZERANK_API_KEY"):
        if not os.getenv(var):
            raise SystemExit(f"Missing required env var: {var}")

    if args.smoke:
        n_queries = 1
        k_sweep = (100, HYBRID_K)
        repeats = 1
        print("SMOKE: 1 query, K∈{100,2000}, 1 repeat")
    else:
        n_queries = args.n_queries
        k_sweep = _parse_k_sweep(args.k_sweep)
        repeats = args.repeats

    if args.all_datasets:
        slugs = [s for s in DATASETS]
    else:
        slugs = [args.dataset]

    headers = get_embedding_headers(EMBEDDING_MODEL)
    weaviate_client = await _connect_weaviate(headers)
    providers = Providers()
    await providers.open()

    payloads: dict[str, dict] = {}
    try:
        for slug in slugs:
            try:
                payload = await measure_dataset(
                    slug,
                    providers,
                    weaviate_client,
                    n_queries=n_queries,
                    seed=args.seed,
                    k_sweep=k_sweep,
                    repeats=repeats,
                    hybrid_alpha=args.hybrid_alpha,
                    network_location=args.network_location,
                    priced_date=args.priced_date,
                )
            except FileNotFoundError as e:
                if args.all_datasets:
                    print(f"  [skip] {slug}: {e}")
                    continue
                raise
            write_dataset_json(payload)
            payloads[slug] = payload
    finally:
        await providers.close()
        await weaviate_client.close()

    if args.all_datasets and payloads:
        write_latency_md(payloads)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS.keys()),
        default="robotics",
        help="Dataset slug (default: robotics). Ignored with --all-datasets.",
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Loop every dataset, write each latency_{dataset}.json, then "
             "emit results/latency/LATENCY.md.",
    )
    parser.add_argument("--n-queries", type=int, default=DEFAULT_N_QUERIES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--k-sweep",
        type=str,
        default=",".join(str(k) for k in DEFAULT_K_SWEEP),
        help="Comma-separated K values for measurement 1.",
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fast wiring check: 1 query, K∈{100,2000}, 1 repeat.",
    )
    parser.add_argument(
        "--hybrid-alpha",
        type=float,
        default=None,
        help="Hybrid alpha. Default None = Weaviate hybrid default (matches "
             "the experiment's retrieval). Pass e.g. 0.5 to override.",
    )
    parser.add_argument(
        "--network-location",
        type=str,
        default="experiment-host (override with --network-location)",
        help="Free-text network location recorded for reproducibility "
             "(these are hosted APIs, so latency is end-to-end from here).",
    )
    parser.add_argument(
        "--priced-date",
        type=str,
        default=datetime.date.today().isoformat(),
        help="Date stamp recorded in every output file (default: today).",
    )
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
