"""Ports — the boundary interfaces of the hexagon.

The application core talks to the outside world only through these shapes.
They are structural (typing.Protocol): adapters satisfy them by matching the
shape, no inheritance required, and the pydantic models in
src.adapters.retrieval.models serve as the concrete item types.

Driving port (how the evaluation harness invokes the core):

    SearchAgent — what `query_agent_benchmarking.run_search_eval` calls.
        Implementations: src.adapters.qab.RetrieverSearchAgent (live
        retrieval+rerank), src.application.collect.CollectScoresAgent
        (live collection into the score cache), and
        src.application.derived.DerivedSearchAgent (cache-only derivation).

Driven ports (what the core needs from infrastructure):

    RerankFn   — a provider rerank callable, as produced by the
                 make_*_reranker factories in src.adapters.retrieval.providers
                 (Cohere / Voyage / ZeroEntropy behind chunking + byte budgets).
    Retriever  — first-stage retrieval; implemented by
                 src.adapters.retrieval.base_retriever.BaseRetriever
                 (Weaviate hybrid search).
    ScoreStore — per-query reranker-score persistence; implemented by
                 src.adapters.cache.ScoreCache (resumable JSON snapshots
                 under results/<dataset>/caches/).
"""
from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence, runtime_checkable


@runtime_checkable
class SearchAgent(Protocol):
    """Driving port: the shape qab's run_search_eval drives."""

    def run(self, query: str, tenant: Optional[str] = None) -> list:
        """Return ranked document ids (qab ObjectIDs) for one query."""
        ...

    async def run_async(self, query: str, tenant: Optional[str] = None) -> list:
        ...


class ScoredItem(Protocol):
    """One reranked document, as every provider adapter returns it."""

    index: int
    relevance_score: float


class RerankFn(Protocol):
    """Driven port: one provider's rerank call (already chunked/budgeted)."""

    def __call__(
        self, query: str, documents: Sequence[str], top_k: int
    ) -> Sequence[ScoredItem]:
        ...


class Retriever(Protocol):
    """Driven port: first-stage retrieval returning a response with .sources."""

    def forward(self, query: str) -> Any:
        ...


class ScoreStore(Protocol):
    """Driven port: the per-query score snapshot every derivation reads."""

    metadata: dict
    queries: dict

    def save(self, path: Any) -> None:
        ...
