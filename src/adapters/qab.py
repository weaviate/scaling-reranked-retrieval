"""Adapter for query_agent_benchmarking (qab), the evaluation harness.

Two responsibilities, both about interfacing that one external library:

1. `RetrieverSearchAgent` — wraps a retrieval/ retriever (typically a
   CrossEncoderReranker configured for a single condition) so it satisfies
   the SearchAgent protocol expected by `qab.run_search_eval`.

2. `setup()` — runtime guards applied explicitly by every entry module
   before using qab (never on import of the library, so src stays
   side-effect-free):

   - `ensure_qab_version()` fails fast if the environment holds qab < 0.7
     (pyproject pins >=0.7 and uv.lock locks it, so `uv run` already syncs
     the right version — this guards against invoking a script in a stale
     environment). The version is read from importlib.metadata, NOT
     `qab.__version__`, which is a stale constant that wrongly reports
     "0.5" in the 0.7 release.

   - `patch_qab_loader()` memoizes qab's per-dataset corpus load. qab's
     load_bright re-materializes the entire BRIGHT corpus (~62k docs for
     robotics, ~12s) on every run_search_eval call, and a sweep calls it
     once per condition — memoizing the dispatch loads each dataset at most
     once per process.
"""
from __future__ import annotations

import functools
import os
from typing import Optional

import weaviate
from query_agent_benchmarking import ObjectID

from src.adapters.retrieval.embeddings_registry import get_embedding_headers

from src.config import EMBEDDING_MODEL

_MIN_QAB = "0.7"
_loader_patched = False


def ensure_qab_version() -> None:
    """Enforce query_agent_benchmarking >= 0.7 (see module docstring)."""
    from importlib.metadata import version as _pkg_version

    from packaging.version import Version

    qab_version = _pkg_version("query-agent-benchmarking")
    if Version(qab_version) < Version(_MIN_QAB):
        raise SystemExit(
            f"query-agent-benchmarking>={_MIN_QAB} required, found {qab_version}. "
            "Run via `uv run` to sync the environment to uv.lock."
        )


def patch_qab_loader() -> None:
    """Memoize qab's per-dataset load (idempotent; see module docstring)."""
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
