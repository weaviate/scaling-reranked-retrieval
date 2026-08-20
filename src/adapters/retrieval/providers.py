"""Per-provider reranker adapters (Cohere / Voyage / ZeroEntropy).

Each make_*_reranker factory wraps a provider SDK client into a uniform
callable `(query, documents, top_k) -> List[RerankItem]` that accepts an
arbitrary number of documents; batches beyond the provider's per-call limits
are split transparently by the machinery in retrieval.chunking:

  - Cohere: count-capped only (1000 docs/call).
  - Voyage: count-capped + a 600K-tokens-per-batch budget that isn't
    expressible as a doc count. Handled reactively: the SDK's 400
    "max allowed tokens per submitted batch" error triggers recursive chunk
    halving (needed for long-doc datasets, e.g. bright/economics).
  - ZeroEntropy (zerank): count-capped + a 5MB UTF-8 request-byte cap.
    Handled proactively (byte-budget packing under ZERANK_BYTE_BUDGET, leaving
    headroom for query + JSON escaping) with the same reactive halving as a
    safety net (needed for long-doc datasets, e.g. bright/robotics).
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, List, Literal

from src.adapters.retrieval.chunking import (
    MAX_DOCS_PER_CALL,
    async_byte_budget_chunked_rerank,
    async_chunked_rerank,
    byte_budget_chunked_rerank,
    chunked_rerank,
    make_async_halving_call,
    make_halving_call,
)
from src.adapters.retrieval.models import RerankItem

Provider = Literal["cohere", "voyage", "zerank", "hybrid"]
FusionMethod = Literal["rrf", "rsf"]
RERANKER_PROVIDERS: tuple[str, ...] = ("cohere", "voyage", "zerank")


# --------------------------------------------------------------------------- #
# Overflow-error detection                                                     #
# --------------------------------------------------------------------------- #

# Substring lookup used to detect Voyage's per-batch token-budget overflow.
# Voyage's SDK raises voyageai.error.InvalidRequestError; we match on the
# message rather than the exception type to avoid a hard dependency on the
# SDK's exception hierarchy. The message looks like:
#   "Request to model 'rerank-2.5' failed. The max allowed tokens per
#    submitted batch is 600000. Your batch has 607519 tokens after
#    truncation. Please lower the number of tokens in the batch."
_VOYAGE_TOKEN_LIMIT_MARKER = "max allowed tokens per submitted batch"


def _is_voyage_token_limit_error(exc: BaseException) -> bool:
    return _VOYAGE_TOKEN_LIMIT_MARKER in str(exc).lower()


# ZeroEntropy (zerank) enforces a hard cap on the total UTF-8 byte size of the
# request payload (query + documents + JSON envelope). The default Organization
# limit is 5,000,000 bytes; exceeding it returns a 400 BadRequestError whose
# message contains "UTF-8 bytes in this request". ZERANK_BYTE_BUDGET is a
# conservative fraction of the hard limit, leaving headroom for the query +
# JSON escaping.
ZERANK_BYTE_BUDGET = 4_500_000
_ZERANK_BYTE_LIMIT_MARKER = "utf-8 bytes in this request"


def _is_zerank_byte_limit_error(exc: BaseException) -> bool:
    return _ZERANK_BYTE_LIMIT_MARKER in str(exc).lower()


# --------------------------------------------------------------------------- #
# Voyage TPM pacing                                                            #
# --------------------------------------------------------------------------- #

# Post-call sleep (seconds) for Voyage rerank, paced to stay under the 4M
# tokens-per-minute cap at large retrieved_k. With 1000 docs × ~500 tokens
# per chunk and two chunks per query at retrieved_k=2000, each query sends
# ~1M Voyage tokens in a burst — so a 30 s sleep after the rerank completes
# caps the rate at ~1.8M TPM, well under the 4M cap. Set via
# configure_voyage_post_call_sleep(). Sleep applies only on success; a
# 429-or-other failure raises before the sleep runs.
_voyage_post_call_sleep_seconds: float = 0.0


def configure_voyage_post_call_sleep(seconds: float) -> None:
    """Set the post-call sleep duration for Voyage rerank operations."""
    global _voyage_post_call_sleep_seconds
    _voyage_post_call_sleep_seconds = max(0.0, float(seconds))


# --------------------------------------------------------------------------- #
# Cohere                                                                       #
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Voyage                                                                       #
# --------------------------------------------------------------------------- #


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
        # Pace successive Voyage calls to stay under the 4M-TPM cap. Runs
        # only on success; failures raise before this sleep.
        if _voyage_post_call_sleep_seconds > 0:
            await asyncio.sleep(_voyage_post_call_sleep_seconds)
        return items
    return _fn


# --------------------------------------------------------------------------- #
# ZeroEntropy (zerank)                                                         #
# --------------------------------------------------------------------------- #


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
