"""Two-stage retrieve-then-rerank retriever.

Runs first-stage retrieval via BaseRetriever, then rescores the pool with the
hosted cohere/voyage/zerank rerankers (a single provider, or "hybrid" = all
available fused with RRF/RSF) and returns the reranked top reranked_k.
"""
from typing import Optional, List, Dict

import weaviate

from src.adapters.retrieval.weaviate_database import weaviate_search_tool, async_weaviate_search_tool
from src.adapters.retrieval.models import AgentRAGResponse, RerankerClient
from src.adapters.retrieval.base_retriever import BaseRetriever
from src.adapters.retrieval.providers import Provider, FusionMethod
from src.adapters.retrieval.rerank import ce_rank, async_ce_rank, reorder
from src.adapters.retrieval.truncate_document import truncate_document


class CrossEncoderReranker(BaseRetriever):
    def __init__(
        self,
        collection_name: str,
        target_property_name: str,
        weaviate_client: Optional[weaviate.WeaviateClient] = None,
        reranker_clients: Optional[List[RerankerClient]] = None,
        return_property_name: Optional[str] = None,
        verbose: Optional[bool] = False,
        search_only: Optional[bool] = True,
        retrieved_k: Optional[int] = 50,
        reranked_k: Optional[int] = 20,
        model_name_overrides: Optional[Dict[str, str]] = None,
        provider: Optional[Provider] = None,
        rrf_k: Optional[int] = 60,
        hybrid_weights: Optional[Dict[str, float]] = None,
        fusion_method: FusionMethod = "rrf",
    ):
        super().__init__(
            weaviate_client=weaviate_client,
            collection_name=collection_name,
            target_property_name=target_property_name,
            verbose=verbose,
            search_only=search_only,
            retrieved_k=retrieved_k,
        )
        self.return_property_name = return_property_name
        self.reranker_clients = reranker_clients
        self.reranked_k = int(reranked_k or 20)
        self.model_name_overrides = model_name_overrides or {}
        self.provider = provider
        self.rrf_k = int(rrf_k or 60)
        self.hybrid_weights = hybrid_weights
        self.fusion_method: FusionMethod = fusion_method
        self.verbose = bool(verbose)

    def forward(
        self,
        question: str,
        weaviate_client: Optional[weaviate.WeaviateClient] = None,
        reranker_clients: Optional[List[RerankerClient]] = None,
    ) -> AgentRAGResponse:
        weaviate_client = weaviate_client or self.weaviate_client
        reranker_clients = reranker_clients or self.reranker_clients

        # Retrieve
        sources = weaviate_search_tool(
            weaviate_client=weaviate_client,
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            return_property_name=self.return_property_name,
            retrieved_k=self.retrieved_k,
        )

        if self.verbose:
            print(f"Retrieved {len(sources)} documents")

        all_clients = list(reranker_clients) if reranker_clients else []

        # Rerank
        docs = [truncate_document(s.content, 500) for s in sources]
        items = ce_rank(
            query=question,
            documents=docs,
            top_k=self.reranked_k,
            clients=all_clients,
            provider=self.provider,
            model_name_overrides=self.model_name_overrides,
            rrf_k=self.rrf_k,
            hybrid_weights=self.hybrid_weights,
            fusion_method=self.fusion_method,
            verbose=self.verbose,
        )

        reranked = reorder(items, sources)
        if self.verbose:
            print(f"Reranked: Returning {len(reranked)} documents")

        return AgentRAGResponse(
            final_answer="",
            sources=reranked,
            searches=[question],
            aggregations=None,
            usage={},
        )

    async def aforward(
        self,
        question: str,
        weaviate_async_client: Optional[weaviate.WeaviateAsyncClient] = None,
        reranker_clients: Optional[List[RerankerClient]] = None,
    ) -> AgentRAGResponse:
        weaviate_async_client = weaviate_async_client or self.weaviate_async_client
        reranker_clients = reranker_clients or self.reranker_clients

        # Retrieve
        sources = await async_weaviate_search_tool(
            weaviate_async_client=weaviate_async_client,
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            return_property_name=self.return_property_name,
            retrieved_k=self.retrieved_k,
        )

        if self.verbose:
            print(f"Retrieved {len(sources)} documents (async)")

        all_clients = list(reranker_clients) if reranker_clients else []

        # Rerank
        docs = [truncate_document(s.content, 500) for s in sources]
        items = await async_ce_rank(
            query=question,
            documents=docs,
            top_k=self.reranked_k,
            clients=all_clients,
            provider=self.provider,
            model_name_overrides=self.model_name_overrides,
            rrf_k=self.rrf_k,
            hybrid_weights=self.hybrid_weights,
            fusion_method=self.fusion_method,
            verbose=self.verbose,
        )

        reranked = reorder(items, sources)
        if self.verbose:
            print(f"Reranked: Returning {len(reranked)} documents (async)")

        return AgentRAGResponse(
            final_answer="",
            sources=reranked,
            searches=[question],
            aggregations=None,
            usage={},
        )
