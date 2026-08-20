"""Provider-agnostic chunked-rerank machinery.

Hosted rerank APIs cap each request by document count (all three providers:
1000 docs/call) and sometimes by payload size (Voyage: 600K tokens/batch,
ZeroEntropy: 5MB UTF-8 bytes/request). Because cross-encoder rerankers score
each (query, doc) pair independently of batch composition, splitting a batch
across calls and merging by score is exact (not an approximation).

This module holds the generic machinery only — no provider names, models, or
error-message knowledge (those live in retrieval.providers):

  - count-based chunking:      chunked_rerank / async_chunked_rerank
  - byte-budget chunking:      byte_budget_chunks + byte_budget_chunked_rerank
                               / async_byte_budget_chunked_rerank
  - reactive halving retry:    make_halving_call / make_async_halving_call
                               (parameterized by an is-overflow predicate)
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, List

from src.adapters.retrieval.models import RerankItem

# Per-provider hard limit on documents per rerank request. When we send more
# than this in a single call, the make_*_reranker wrappers transparently split
# the documents into chunks, rerank each chunk, and merge by score.
MAX_DOCS_PER_CALL: dict[str, int] = {
    "cohere": 1000,
    "voyage": 1000,
    "zerank": 1000,
}


def _offset_items(items: List[RerankItem], start: int) -> List[RerankItem]:
    """Re-index a chunk's items back into the original document list."""
    return [
        RerankItem(index=it.index + start, relevance_score=it.relevance_score)
        for it in items
    ]


def _merge_chunks(
    chunk_results: List[tuple[int, List[RerankItem]]], top_k: int
) -> List[RerankItem]:
    """Merge (start_index, items) chunk results: re-index, sort, cut to top_k."""
    all_items: List[RerankItem] = []
    for start, items in chunk_results:
        all_items.extend(_offset_items(items, start))
    all_items.sort(key=lambda x: x.relevance_score, reverse=True)
    return all_items[: min(top_k, len(all_items))]


def count_chunks(documents: List[str], max_per_call: int) -> List[tuple[int, List[str]]]:
    """Split documents into (start_index, chunk) pairs of at most max_per_call."""
    chunks: List[tuple[int, List[str]]] = []
    for start in range(0, len(documents), max_per_call):
        end = min(start + max_per_call, len(documents))
        chunks.append((start, documents[start:end]))
    return chunks


def byte_budget_chunks(
    documents: List[str], max_per_call: int, byte_budget: int
) -> List[tuple[int, List[str]]]:
    """Greedily pack documents into (start_index, chunk) pairs bounded by both a
    max doc count and a cumulative UTF-8 byte budget. A document larger than the
    whole budget gets its own chunk (handled reactively / re-raised downstream).
    Order is preserved, so start_index + local index recovers the original."""
    chunks: List[tuple[int, List[str]]] = []
    cur: List[str] = []
    cur_bytes = 0
    cur_start = 0
    for i, doc in enumerate(documents):
        dbytes = len(doc.encode("utf-8"))
        if cur and (len(cur) >= max_per_call or cur_bytes + dbytes > byte_budget):
            chunks.append((cur_start, cur))
            cur = []
            cur_bytes = 0
            cur_start = i
        cur.append(doc)
        cur_bytes += dbytes
    if cur:
        chunks.append((cur_start, cur))
    return chunks


# --------------------------------------------------------------------------- #
# Count-based chunked rerank                                                   #
# --------------------------------------------------------------------------- #


def chunked_rerank(
    call_one_chunk: Callable[[str, List[str], int], List[RerankItem]],
    query: str,
    documents: List[str],
    top_k: int,
    max_per_call: int,
) -> List[RerankItem]:
    """Synchronously rerank documents via possibly multiple chunked API calls."""
    if len(documents) <= max_per_call:
        return call_one_chunk(query, documents, min(top_k, len(documents)))

    results = [
        (start, call_one_chunk(query, chunk, len(chunk)))
        for start, chunk in count_chunks(documents, max_per_call)
    ]
    return _merge_chunks(results, top_k)


async def async_chunked_rerank(
    call_one_chunk: Callable[[str, List[str], int], Any],
    query: str,
    documents: List[str],
    top_k: int,
    max_per_call: int,
) -> List[RerankItem]:
    """Asynchronously rerank documents via concurrent chunked API calls."""
    if len(documents) <= max_per_call:
        return await call_one_chunk(query, documents, min(top_k, len(documents)))

    chunks = count_chunks(documents, max_per_call)
    chunk_items = await asyncio.gather(
        *(call_one_chunk(query, docs, len(docs)) for _, docs in chunks)
    )
    return _merge_chunks(
        [(start, items) for (start, _), items in zip(chunks, chunk_items)], top_k
    )


# --------------------------------------------------------------------------- #
# Byte-budget chunked rerank (proactive splitting for payload-capped APIs)     #
# --------------------------------------------------------------------------- #


def byte_budget_chunked_rerank(
    call_one_chunk: Callable[[str, List[str], int], List[RerankItem]],
    query: str,
    documents: List[str],
    top_k: int,
    max_per_call: int,
    byte_budget: int,
) -> List[RerankItem]:
    """Sync byte-budget-aware chunked rerank."""
    chunks = byte_budget_chunks(documents, max_per_call, byte_budget)
    if len(chunks) == 1:
        _, docs = chunks[0]
        return call_one_chunk(query, docs, min(top_k, len(docs)))

    results = [(start, call_one_chunk(query, docs, len(docs))) for start, docs in chunks]
    return _merge_chunks(results, top_k)


async def async_byte_budget_chunked_rerank(
    call_one_chunk: Callable[[str, List[str], int], Any],
    query: str,
    documents: List[str],
    top_k: int,
    max_per_call: int,
    byte_budget: int,
) -> List[RerankItem]:
    """Async byte-budget-aware chunked rerank; chunks run concurrently."""
    chunks = byte_budget_chunks(documents, max_per_call, byte_budget)
    if len(chunks) == 1:
        _, docs = chunks[0]
        return await call_one_chunk(query, docs, min(top_k, len(docs)))

    chunk_items = await asyncio.gather(
        *(call_one_chunk(query, docs, len(docs)) for _, docs in chunks)
    )
    return _merge_chunks(
        [(start, items) for (start, _), items in zip(chunks, chunk_items)], top_k
    )


# --------------------------------------------------------------------------- #
# Reactive halving retry (for length-dependent limits the proactive split      #
# can't fully predict: token counts, JSON-escaping inflation, …)               #
# --------------------------------------------------------------------------- #


def make_halving_call(
    call_chunk: Callable[[str, List[str], int], List[RerankItem]],
    is_overflow_error: Callable[[BaseException], bool],
) -> Callable[[str, List[str], int], List[RerankItem]]:
    """Wrap call_chunk so a payload-overflow error recursively halves the chunk
    and retries each half. Returns all per-doc scores with indices in
    [0, len(docs)); the outer chunked-rerank does the final sort + top_k cut.
    Single-doc chunks that still overflow are re-raised so the failure surfaces
    (a single (query, doc) pair exceeding the budget is not something splitting
    can solve)."""

    def safe_call_chunk(query: str, docs: List[str], top_n: int) -> List[RerankItem]:
        try:
            return call_chunk(query, docs, top_n)
        except Exception as e:
            if not is_overflow_error(e) or len(docs) <= 1:
                raise
            mid = len(docs) // 2
            left = safe_call_chunk(query, docs[:mid], mid)
            right = safe_call_chunk(query, docs[mid:], len(docs) - mid)
            return left + _offset_items(right, mid)

    return safe_call_chunk


def make_async_halving_call(
    call_chunk: Callable[[str, List[str], int], Any],
    is_overflow_error: Callable[[BaseException], bool],
) -> Callable[[str, List[str], int], Any]:
    """Async counterpart to make_halving_call. The two halves are awaited
    concurrently — they're independent and the failed request that triggered
    the split was a 400 (not rate-limited), so the concurrent burst doesn't
    compound any rate pressure."""

    async def safe_call_chunk(query: str, docs: List[str], top_n: int) -> List[RerankItem]:
        try:
            return await call_chunk(query, docs, top_n)
        except Exception as e:
            if not is_overflow_error(e) or len(docs) <= 1:
                raise
            mid = len(docs) // 2
            left, right = await asyncio.gather(
                safe_call_chunk(query, docs[:mid], mid),
                safe_call_chunk(query, docs[mid:], len(docs) - mid),
            )
            return left + _offset_items(right, mid)

    return safe_call_chunk
