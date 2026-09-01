"""Adapter for query_agent_benchmarking (qab): the SearchAgent bridge plus
runtime guards (setup() = version check + corpus-loader memoization), applied
explicitly by entry modules so import stays side-effect-free.
"""
from __future__ import annotations

import functools
import os
from typing import Optional

import weaviate
from query_agent_benchmarking import ObjectID

from scaling_reranked_retrieval.adapters.retrieval.embeddings_registry import get_embedding_headers

from scaling_reranked_retrieval.config import EMBEDDING_MODEL

_MIN_QAB = "0.7"
_loader_patched = False


def ensure_qab_version() -> None:
    """Enforce qab >= 0.7 via importlib.metadata (qab.__version__ is a stale constant)."""
    from importlib.metadata import version as _pkg_version

    from packaging.version import Version

    qab_version = _pkg_version("query-agent-benchmarking")
    if Version(qab_version) < Version(_MIN_QAB):
        raise SystemExit(
            f"query-agent-benchmarking>={_MIN_QAB} required, found {qab_version}. "
            "Run via `uv run` to sync the environment to uv.lock."
        )


def patch_qab_loader() -> None:
    """Memoize qab's per-dataset corpus load (idempotent; qab reloads it per run_search_eval call)."""
    global _loader_patched
    if _loader_patched:
        return
    import query_agent_benchmarking.internal.adapters.dataset as _qab_ds

    _qab_ds.load_search_dataset = functools.cache(_qab_ds.load_search_dataset)
    _loader_patched = True


def setup() -> None:
    """Version guard + loader memoization; call once at entry-point start."""
    ensure_qab_version()
    patch_qab_loader()


class RetrieverSearchAgent:
    """Adapts any retrieval/ retriever to the SearchAgent protocol."""

    def __init__(self, retriever, embedding_model: str = EMBEDDING_MODEL):
        self.retriever = retriever
        self.embedding_model = embedding_model
        self._async_client: Optional[weaviate.WeaviateAsyncClient] = None

    def run(self, query: str, tenant: Optional[str] = None) -> list[ObjectID]:
        response = self.retriever.forward(query)
        return [ObjectID(object_id=s.object_id) for s in response.sources]

    async def run_async(
        self, query: str, tenant: Optional[str] = None
    ) -> list[ObjectID]:
        response = await self.retriever.aforward(
            query, weaviate_async_client=self._async_client
        )
        return [ObjectID(object_id=s.object_id) for s in response.sources]

    async def initialize_async(self) -> None:
        headers = get_embedding_headers(self.embedding_model)
        self._async_client = weaviate.use_async_with_weaviate_cloud(
            cluster_url=os.environ["WEAVIATE_URL"],
            auth_credentials=weaviate.auth.AuthApiKey(os.environ["WEAVIATE_API_KEY"]),
            headers=headers,
            skip_init_checks=True,
        )
        await self._async_client.connect()

    async def close_async(self) -> None:
        if self._async_client is not None:
            await self._async_client.close()
