"""Reranking dispatch: adapter registry, provider selection, fusion, entry points.

`ce_rank` / `async_ce_rank` are the top-level calls the retrievers use: they
turn a list of RerankerClient wrappers into per-provider rerank callables
(retrieval.providers), pick a provider (a single one, or "hybrid" = run all
available and fuse with RRF/RSF), and return reranked RerankItems. `reorder`
maps those items back onto the retrieved ObjectFromDB sources.
"""
from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable, Dict, List, Optional

from src.adapters.retrieval.models import ObjectFromDB, RerankerClient, RerankItem
from src.adapters.retrieval.providers import (
    RERANKER_PROVIDERS,
    FusionMethod,
    Provider,
    make_async_cohere_reranker,
    make_async_voyage_reranker,
    make_async_zerank_reranker,
    make_cohere_reranker,
    make_voyage_reranker,
    make_zerank_reranker,
)


def _get_model_name(provider: str, overrides: Optional[Dict[str, str]], default: str) -> str:
    return overrides.get(provider, default) if overrides else default


def _make_adapters(
    clients: Optional[List[RerankerClient]],
    overrides: Optional[Dict[str, str]],
) -> Dict[str, Callable]:
    """Create sync reranker functions from clients."""
    adapters = {}
    if not clients:
        return adapters

    for rc in clients:
        if rc.name == "cohere":
            adapters["cohere"] = make_cohere_reranker(rc.client, _get_model_name("cohere", overrides, "rerank-v3.5"))
        elif rc.name == "voyage":
            adapters["voyage"] = make_voyage_reranker(rc.client, _get_model_name("voyage", overrides, "rerank-2.5"))
        elif rc.name == "zerank":
            # Only the sync ZeroEntropy client goes into sync adapters; the
            # async AsyncZeroEntropy client doesn't expose a callable .rerank
            # at the top level, so guard against picking it up here.
            if hasattr(rc.client, "models") and not inspect.iscoroutinefunction(
                rc.client.models.rerank
            ):
                adapters["zerank"] = make_zerank_reranker(
                    rc.client, _get_model_name("zerank", overrides, "zerank-2")
                )
        elif callable(rc.client) and not hasattr(rc.client, 'rerank'):
            # Custom callable reranker (already wrapped)
            adapters[rc.name] = rc.client

    return adapters


def _make_async_adapters(
    clients: Optional[List[RerankerClient]],
    overrides: Optional[Dict[str, str]],
) -> Dict[str, Callable]:
    """Create async reranker functions from clients."""
    adapters = {}
    if not clients:
        return adapters

    for rc in clients:
        if rc.name == "zerank":
            # Only the AsyncZeroEntropy client gets a native async adapter.
            if hasattr(rc.client, "models") and inspect.iscoroutinefunction(
                rc.client.models.rerank
            ):
                adapters["zerank"] = make_async_zerank_reranker(
                    rc.client, _get_model_name("zerank", overrides, "zerank-2")
                )
        elif callable(rc.client) and inspect.iscoroutinefunction(rc.client):
            # Custom async callable reranker (already wrapped)
            adapters[rc.name] = rc.client
        elif hasattr(rc.client, 'rerank') and inspect.iscoroutinefunction(rc.client.rerank):
            if rc.name == "cohere":
                adapters["cohere"] = make_async_cohere_reranker(rc.client, _get_model_name("cohere", overrides, "rerank-v3.5"))
            elif rc.name == "voyage":
                adapters["voyage"] = make_async_voyage_reranker(rc.client, _get_model_name("voyage", overrides, "rerank-2.5"))

    return adapters


def _pick_provider(requested: Optional[Provider], available: Dict[str, Any]) -> Provider:
    """Auto-select provider based on what's available."""
    if requested:
        return requested

    present = [p for p in RERANKER_PROVIDERS if p in available]

    if len(present) > 1:
        return "hybrid"
    if len(present) == 1:
        return present[0]  # type: ignore[return-value]

    return "cohere"  # Fallback


def _rerank_single(provider: str, query: str, docs: List[str], top_k: int, rerankers: Dict) -> List[RerankItem]:
    """Rerank with single provider."""
    return rerankers[provider](query, docs, top_k)


def _apply_fusion(
    method: FusionMethod,
    results: Dict[str, List[RerankItem]],
    top_k: int,
    rrf_k: int,
    weights: Optional[Dict[str, float]],
) -> List[RerankItem]:
    from src.adapters.retrieval.rrf import fuse_rrf
    from src.adapters.retrieval.rsf import fuse_rsf

    if method == "rsf":
        return fuse_rsf(results, top_k, weights=weights)
    return fuse_rrf(results, top_k, rrf_k=rrf_k, weights=weights)


def rerank(
    provider: Provider,
    query: str,
    documents: List[str],
    top_k: int,
    rerankers: Dict[str, Callable],
    rrf_k: int = 60,
    hybrid_weights: Optional[Dict[str, float]] = None,
    fusion_method: FusionMethod = "rrf",
) -> List[RerankItem]:
    """Sync reranking."""
    if provider in RERANKER_PROVIDERS:
        return _rerank_single(provider, query, documents, top_k, rerankers)

    # Hybrid mode - run all available providers
    results = {}
    for p in RERANKER_PROVIDERS:
        if p in rerankers:
            try:
                results[p] = rerankers[p](query, documents, top_k)
            except Exception:
                results[p] = []

    return _apply_fusion(fusion_method, results, top_k, rrf_k, hybrid_weights)


async def async_rerank(
    provider: Provider,
    query: str,
    documents: List[str],
    top_k: int,
    async_rerankers: Optional[Dict[str, Callable]] = None,
    rerankers: Optional[Dict[str, Callable]] = None,
    rrf_k: int = 60,
    hybrid_weights: Optional[Dict[str, float]] = None,
    fusion_method: FusionMethod = "rrf",
) -> List[RerankItem]:
    """Async reranking."""
    async_rerankers = async_rerankers or {}
    rerankers = rerankers or {}

    async def _run(p: str) -> List[RerankItem]:
        if p in async_rerankers:
            return await async_rerankers[p](query, documents, top_k)
        if p in rerankers:
            return await asyncio.to_thread(rerankers[p], query, documents, top_k)
        return []

    if provider in RERANKER_PROVIDERS:
        return await _run(provider)

    # Hybrid mode - run all available providers concurrently
    tasks = {p: asyncio.create_task(_run(p)) for p in RERANKER_PROVIDERS}
    results = {p: await task for p, task in tasks.items()}

    return _apply_fusion(fusion_method, results, top_k, rrf_k, hybrid_weights)


def ce_rank(
    query: str,
    documents: List[str],
    top_k: int,
    clients: Optional[List[RerankerClient]] = None,
    provider: Optional[Provider] = None,
    model_name_overrides: Optional[Dict[str, str]] = None,
    rrf_k: int = 60,
    hybrid_weights: Optional[Dict[str, float]] = None,
    fusion_method: FusionMethod = "rrf",
    verbose: bool = False,
) -> List[RerankItem]:
    """Sync rerank documents."""
    adapters = _make_adapters(clients, model_name_overrides)
    eff_provider = _pick_provider(provider, adapters)
    return rerank(
        eff_provider, query, documents, top_k, adapters,
        rrf_k, hybrid_weights, fusion_method,
    )


async def async_ce_rank(
    query: str,
    documents: List[str],
    top_k: int,
    clients: Optional[List[RerankerClient]] = None,
    provider: Optional[Provider] = None,
    model_name_overrides: Optional[Dict[str, str]] = None,
    rrf_k: int = 60,
    hybrid_weights: Optional[Dict[str, float]] = None,
    fusion_method: FusionMethod = "rrf",
    verbose: bool = False,
) -> List[RerankItem]:
    """Async rerank documents."""
    sync_adapters = _make_adapters(clients, model_name_overrides)
    async_adapters = _make_async_adapters(clients, model_name_overrides)
    all_adapters = {**sync_adapters, **async_adapters}
    eff_provider = _pick_provider(provider, all_adapters)

    return await async_rerank(
        eff_provider, query, documents, top_k,
        async_adapters, sync_adapters, rrf_k, hybrid_weights, fusion_method,
    )


def reorder(items: List[RerankItem], sources: List[ObjectFromDB]) -> List[ObjectFromDB]:
    """Reorder sources and update ranks/scores."""
    out = []
    for new_rank, item in enumerate(items, start=1):
        if 0 <= item.index < len(sources):
            orig = sources[item.index]
            out.append(ObjectFromDB(
                object_id=orig.object_id,
                content=orig.content,
                relevance_rank=new_rank,
                relevance_score=item.relevance_score,
                vector=orig.vector,
                source_query=orig.source_query,
            ))
    return out
