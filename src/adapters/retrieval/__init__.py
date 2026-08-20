"""Retrieval infrastructure: Weaviate search + hosted-reranker callers.

    models.py                 pydantic data models (ObjectFromDB, RerankItem, …)
    clients.py                provider client factories (weaviate/cohere/voyage/zerank)
    chunking.py               provider-agnostic chunked-rerank machinery:
                              count/byte-budget chunking + reactive halving retry
    providers.py              cohere/voyage/zerank rerank adapters, per-provider
                              limits (token/byte budgets), Voyage TPM pacing
    rerank.py                 adapter registry, provider selection, fusion dispatch,
                              ce_rank / async_ce_rank / reorder entry points
    rrf.py / rsf.py           Reciprocal Rank / Relative Score Fusion
    truncate_document.py      word-cap document truncation for rerank payloads
    embeddings_registry.py    embedding-provider auth headers for Weaviate
    database.py               SearchBackend protocol + WeaviateBackend
    weaviate_database.py      weaviate hybrid/bm25/vector search tools
    base_retriever.py         first-stage hybrid retriever
    cross_encoder_reranker.py two-stage retrieve-then-rerank retriever
"""

from src.adapters.retrieval.base_retriever import BaseRetriever
from src.adapters.retrieval.cross_encoder_reranker import CrossEncoderReranker

__all__ = ["BaseRetriever", "CrossEncoderReranker"]
