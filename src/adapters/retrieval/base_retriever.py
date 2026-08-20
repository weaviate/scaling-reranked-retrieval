"""First-stage hybrid-search retriever over Weaviate.

``forward``/``aforward`` run one hybrid (BM25 + dense) search and return an
AgentRAGResponse whose ``.sources`` hold the retrieved pool.
``pyversity``/``numpy`` are imported lazily inside ``_maybe_diversify`` so
the diversification path (unused by the experiment, ``diversity_weight=0``)
doesn't force the dependency.
"""
from __future__ import annotations

from typing import Any, Optional, Union, cast

import weaviate

from src.adapters.retrieval.database import (
    AsyncSearchBackend,
    SearchBackend,
    WeaviateBackend,
)
from src.adapters.retrieval.weaviate_database import (
    async_weaviate_search_tool,
    weaviate_search_tool,
)
from src.adapters.retrieval.models import AgentRAGResponse
from src.adapters.retrieval.embeddings_registry import get_embedding_headers

AnyBackend = Union[SearchBackend, AsyncSearchBackend, WeaviateBackend]


class BaseRetriever:
    def __init__(
        self,
        collection_name: str,
        weaviate_client: Optional[weaviate.WeaviateClient] = None,
        target_property_name: Optional[str] = "content",
        verbose: Optional[bool] = True,
        search_only: Optional[bool] = True,
        embedding_model: Optional[str] = None,
        retrieved_k: Optional[int] = 20,
        diversity_weight: Optional[float] = 0,
        search_type: str = "hybrid",
        hybrid_alpha: Optional[float] = None,
        weaviate_async_client: Optional[weaviate.WeaviateAsyncClient] = None,
        backend: Optional[AnyBackend] = None,
    ) -> None:
        self.collection_name = collection_name

        # Only treat the client slots as a backend source when they hold real
        # clients (callers may pass None through).
        sync_client = (
            weaviate_client
            if isinstance(weaviate_client, weaviate.WeaviateClient)
            else None
        )
        async_client = (
            weaviate_async_client
            if isinstance(weaviate_async_client, weaviate.WeaviateAsyncClient)
            else None
        )
        self.weaviate_client = sync_client
        self.weaviate_async_client = async_client

        if backend is None and (sync_client is not None or async_client is not None):
            backend = WeaviateBackend(
                sync_client=sync_client,
                async_client=async_client,
            )
        self.backend: Optional[AnyBackend] = backend

        self.target_property_name: str = target_property_name or "content"
        self.verbose: bool = bool(verbose) if verbose is not None else True
        self.search_only: bool = bool(search_only) if search_only is not None else True
        self.embedding_model = embedding_model
        self.retrieved_k: int = retrieved_k if retrieved_k is not None else 20
        self.diversity_weight: float = diversity_weight or 0.0
        self.search_type = search_type
        self.hybrid_alpha = hybrid_alpha

    @staticmethod
    def _merge_usage(*usages: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
        merged: dict[str, dict[str, int]] = {}
        for usage in usages:
            if usage is None:
                continue
            for lm_id, stats in usage.items():
                bucket = merged.setdefault(
                    lm_id, {"prompt_tokens": 0, "completion_tokens": 0}
                )
                bucket["prompt_tokens"] += stats.get("prompt_tokens", 0)
                bucket["completion_tokens"] += stats.get("completion_tokens", 0)
        return merged

    def get_embedding_headers(self) -> dict[str, str]:
        if self.embedding_model is None:
            return {}
        return get_embedding_headers(self.embedding_model)

    def _resolve_search_k(self) -> int:
        return self.retrieved_k * 2 if self.diversity_weight > 0 else self.retrieved_k

    def _maybe_diversify(self, sources: list, retrieved_k: int) -> list:
        if self.diversity_weight <= 0 or not sources:
            return sources
        import numpy as np
        from pyversity import Strategy, diversify

        vectors = np.array([source.vector for source in sources])
        scores = np.array([source.relevance_score for source in sources])
        diversified_result = diversify(
            embeddings=vectors,
            scores=scores,
            k=retrieved_k,
            strategy=Strategy.MMR,
            diversity=self.diversity_weight,
        )
        return [sources[i] for i in diversified_result.indices]

    def forward(
        self,
        question: str,
        weaviate_client: Optional[weaviate.WeaviateClient] = None,
    ) -> AgentRAGResponse:
        retrieved_k = self._resolve_search_k()
        client = weaviate_client or self.weaviate_client

        if self.backend is not None and weaviate_client is None:
            sources = cast(Any, self.backend).search(
                query=question,
                collection_name=self.collection_name,
                target_property_name=self.target_property_name,
                retrieved_k=retrieved_k,
                return_vector=True,
                return_score=True,
                search_type=self.search_type,
                hybrid_alpha=self.hybrid_alpha,
            )
        else:
            sources = weaviate_search_tool(
                query=question,
                collection_name=self.collection_name,
                target_property_name=self.target_property_name,
                retrieved_k=retrieved_k,
                weaviate_client=client,
                return_vector=True,
                return_score=True,
                search_type=self.search_type,
                hybrid_alpha=self.hybrid_alpha,
            )

        sources = self._maybe_diversify(sources, retrieved_k)

        if self.verbose:
            print(f"\033[96m Returning {len(sources)} Sources!\033[0m")

        return AgentRAGResponse(
            final_answer="",
            sources=sources,
            searches=[question],
            usage={},
        )

    async def aforward(
        self,
        question: str,
        weaviate_async_client: Optional[weaviate.WeaviateAsyncClient] = None,
    ) -> AgentRAGResponse:
        retrieved_k = self._resolve_search_k()
        async_client = weaviate_async_client or self.weaviate_async_client

        if self.backend is not None and weaviate_async_client is None:
            sources = await cast(Any, self.backend).asearch(
                query=question,
                collection_name=self.collection_name,
                target_property_name=self.target_property_name,
                retrieved_k=retrieved_k,
                return_vector=True,
                return_score=True,
                search_type=self.search_type,
                hybrid_alpha=self.hybrid_alpha,
            )
        else:
            sources = await async_weaviate_search_tool(
                query=question,
                collection_name=self.collection_name,
                target_property_name=self.target_property_name,
                retrieved_k=retrieved_k,
                weaviate_async_client=async_client,
                return_vector=True,
                return_score=True,
                search_type=self.search_type,
                hybrid_alpha=self.hybrid_alpha,
            )

        sources = self._maybe_diversify(sources, retrieved_k)

        if self.verbose:
            print(f"\033[96m Returning {len(sources)} Sources!\033[0m")

        return AgentRAGResponse(
            final_answer="",
            sources=sources,
            searches=[question],
            usage={},
        )
