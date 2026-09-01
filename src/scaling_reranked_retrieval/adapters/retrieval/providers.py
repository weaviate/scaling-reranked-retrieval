"""Per-provider reranker adapters (Cohere / Voyage / ZeroEntropy).

Provider API limits handled via retrieval.chunking: all are capped at 1000
docs/call; Voyage adds a 600K-tokens-per-batch budget (reactive halving);
ZeroEntropy adds a 5MB UTF-8 request cap (proactive byte-budget packing plus
reactive halving).
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, List, Literal

from scaling_reranked_retrieval.adapters.retrieval.chunking import (
    MAX_DOCS_PER_CALL,
    async_byte_budget_chunked_rerank,
    async_chunked_rerank,
    byte_budget_chunked_rerank,
    chunked_rerank,
    make_async_halving_call,
    make_halving_call,
)
from scaling_reranked_retrieval.adapters.retrieval.models import RerankItem

Provider = Literal["cohere", "voyage", "zerank", "hybrid"]
FusionMethod = Literal["rrf", "rsf"]
RERANKER_PROVIDERS: tuple[str, ...] = ("cohere", "voyage", "zerank")


# Voyage token-budget overflow is detected by message substring (not exception
# type) to avoid depending on the SDK's exception hierarchy.
_VOYAGE_TOKEN_LIMIT_MARKER = "max allowed tokens per submitted batch"


def _is_voyage_token_limit_error(exc: BaseException) -> bool:
    return _VOYAGE_TOKEN_LIMIT_MARKER in str(exc).lower()


# ZeroEntropy hard-caps request payloads at 5,000,000 UTF-8 bytes; the budget
# leaves headroom for the query + JSON escaping.
ZERANK_BYTE_BUDGET = 4_500_000
_ZERANK_BYTE_LIMIT_MARKER = "utf-8 bytes in this request"


def _is_zerank_byte_limit_error(exc: BaseException) -> bool:
    return _ZERANK_BYTE_LIMIT_MARKER in str(exc).lower()


# Post-call sleep (seconds) pacing Voyage rerank under its 4M tokens-per-minute
# cap; applies only on success (failures raise before the sleep).
_voyage_post_call_sleep_seconds: float = 0.0


def configure_voyage_post_call_sleep(seconds: float) -> None:
    """Set the post-call sleep duration for Voyage rerank operations."""
    global _voyage_post_call_sleep_seconds
    _voyage_post_call_sleep_seconds = max(0.0, float(seconds))


def make_cohere_reranker(client: Any, model: str = "rerank-v3.5") -> Callable:
    def call_chunk(query: str, docs: List[str], top_n: int) -> List[RerankItem]:
        res = client.rerank(model=model, query=query, documents=docs, top_n=top_n)
        return [RerankItem(index=r.index, relevance_score=float(r.relevance_score)) for r in res.results]

    def _fn(query: str, documents: List[str], top_k: int) -> List[RerankItem]:
        return chunked_rerank(call_chunk, query, documents, top_k, MAX_DOCS_PER_CALL["cohere"])
    return _fn


def make_async_cohere_reranker(client: Any, model: str = "rerank-v3.5") -> Callable:
    async def call_chunk(query: str, docs: List[str], top_n: int) -> List[RerankItem]:
        res = await client.rerank(model=model, query=query, documents=docs, top_n=top_n)
        return [RerankItem(index=r.index, relevance_score=float(r.relevance_score)) for r in res.results]

    async def _fn(query: str, documents: List[str], top_k: int) -> List[RerankItem]:
        return await async_chunked_rerank(call_chunk, query, documents, top_k, MAX_DOCS_PER_CALL["cohere"])
    return _fn


def make_voyage_reranker(client: Any, model: str = "rerank-2.5") -> Callable:
    def call_chunk(query: str, docs: List[str], top_n: int) -> List[RerankItem]:
        res = client.rerank(query=query, documents=docs, model=model, top_k=top_n)
        return [RerankItem(index=r.index, relevance_score=float(r.relevance_score)) for r in res.results]

    safe_call_chunk = make_halving_call(call_chunk, _is_voyage_token_limit_error)

    def _fn(query: str, documents: List[str], top_k: int) -> List[RerankItem]:
        return chunked_rerank(safe_call_chunk, query, documents, top_k, MAX_DOCS_PER_CALL["voyage"])
    return _fn


def make_async_voyage_reranker(client: Any, model: str = "rerank-2.5") -> Callable:
    async def call_chunk(query: str, docs: List[str], top_n: int) -> List[RerankItem]:
        res = await client.rerank(query=query, documents=docs, model=model, top_k=top_n)
        return [RerankItem(index=r.index, relevance_score=float(r.relevance_score)) for r in res.results]

    safe_call_chunk = make_async_halving_call(call_chunk, _is_voyage_token_limit_error)

    async def _fn(query: str, documents: List[str], top_k: int) -> List[RerankItem]:
        items = await async_chunked_rerank(
            safe_call_chunk, query, documents, top_k, MAX_DOCS_PER_CALL["voyage"]
        )
        # TPM pacing; runs only on success.
        if _voyage_post_call_sleep_seconds > 0:
            await asyncio.sleep(_voyage_post_call_sleep_seconds)
        return items
    return _fn


def make_zerank_reranker(client: Any, model: str = "zerank-2") -> Callable:
    """Sync wrapper for a `zeroentropy.ZeroEntropy` rerank client."""
    def call_chunk(query: str, docs: List[str], top_n: int) -> List[RerankItem]:
        res = client.models.rerank(model=model, query=query, documents=docs, top_n=top_n)
        return [RerankItem(index=r.index, relevance_score=float(r.relevance_score)) for r in res.results]

    safe_call_chunk = make_halving_call(call_chunk, _is_zerank_byte_limit_error)

    def _fn(query: str, documents: List[str], top_k: int) -> List[RerankItem]:
        return byte_budget_chunked_rerank(
            safe_call_chunk, query, documents, top_k,
            MAX_DOCS_PER_CALL["zerank"], ZERANK_BYTE_BUDGET,
        )
    return _fn


def make_async_zerank_reranker(client: Any, model: str = "zerank-2") -> Callable:
    """Async wrapper for a `zeroentropy.AsyncZeroEntropy` rerank client."""
    async def call_chunk(query: str, docs: List[str], top_n: int) -> List[RerankItem]:
        res = await client.models.rerank(model=model, query=query, documents=docs, top_n=top_n)
        return [RerankItem(index=r.index, relevance_score=float(r.relevance_score)) for r in res.results]

    safe_call_chunk = make_async_halving_call(call_chunk, _is_zerank_byte_limit_error)

    async def _fn(query: str, documents: List[str], top_k: int) -> List[RerankItem]:
        return await async_byte_budget_chunked_rerank(
            safe_call_chunk, query, documents, top_k,
            MAX_DOCS_PER_CALL["zerank"], ZERANK_BYTE_BUDGET,
        )
    return _fn
