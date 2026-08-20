"""CollectScoresAgent: one-time real-call score collection at a chosen k.

The ONLY live-API module in src/. Weaviate and the Cohere/Voyage/Zerank
reranker clients are imported lazily inside initialize_async (the only code
path that issues live API calls). Keeping them out of module scope lets
read-only consumers — DerivedSearchAgent, ScoreCache, and every analysis —
import src without pulling in any provider client (a hard requirement of the
agreement analysis: zero reranker-API surface).
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from query_agent_benchmarking import ObjectID

from src.adapters.cache import ScoreCache
from src.config import EMBEDDING_MODEL

if TYPE_CHECKING:
    import weaviate


class CollectScoresAgent:
    """SearchAgent that records full cohere+voyage+zerank rerank scores per query.

    For every new query: hybrid search top retrieved_k → rerank all with each
    provider concurrently → write entry to cache (with atomic replace).

    Returns the hybrid top-20 to satisfy the SearchAgent protocol; the returned
    metrics from this run are not the experiment's reported numbers.
    """

    def __init__(
        self,
        collection_name: str,
        target_property: str,
        retrieved_k: int,
        cache: ScoreCache,
        cache_path: Path,
        cohere_model: str,
        voyage_model: str,
        zerank_model: str,
    ):
        self.collection_name = collection_name
        self.target_property = target_property
        self.retrieved_k = retrieved_k
        self.cache = cache
        self.cache_path = cache_path
        self.cohere_model = cohere_model
        self.voyage_model = voyage_model
        self.zerank_model = zerank_model
        self._async_client: "Optional[weaviate.WeaviateAsyncClient]" = None
        self._cohere = None
        self._voyage = None
        self._zerank = None  # AsyncZeroEntropy
        # Chunking-aware async rerank functions. Populated in initialize_async.
        # They transparently split documents > 1000 across multiple API calls.
        self._cohere_fn = None
        self._voyage_fn = None
        self._zerank_fn = None
        self._save_lock = asyncio.Lock()

    async def initialize_async(self) -> None:
        # Lazy imports: these are the live-API dependencies (Weaviate + the
        # three reranker provider clients). They live here, not at module
        # scope, so that importing this module for derivation/analysis stays
        # provider-client-free. See the note at the top of the file.
        import weaviate

        from src.adapters.retrieval.clients import (
            get_cohere_async_client,
            get_voyage_async_client,
            get_zerank_async_client,
        )
        from src.adapters.retrieval.providers import (
            make_async_cohere_reranker,
            make_async_voyage_reranker,
            make_async_zerank_reranker,
        )
        from src.adapters.retrieval.embeddings_registry import (
            get_embedding_headers,
        )

        headers = get_embedding_headers(EMBEDDING_MODEL)
        self._async_client = weaviate.use_async_with_weaviate_cloud(
            cluster_url=os.environ["WEAVIATE_URL"],
            auth_credentials=weaviate.auth.AuthApiKey(os.environ["WEAVIATE_API_KEY"]),
            headers=headers,
            skip_init_checks=True,
        )
        await self._async_client.connect()
        self._cohere = get_cohere_async_client().client
        self._voyage = get_voyage_async_client().client
        self._zerank = get_zerank_async_client().client
        # Wrap each client in a chunking-aware callable so retrieved_k > the
        # provider's per-call limit (1000 for all three) is handled
        # transparently across multiple API calls.
        self._cohere_fn = make_async_cohere_reranker(self._cohere, self.cohere_model)
        self._voyage_fn = make_async_voyage_reranker(self._voyage, self.voyage_model)
        self._zerank_fn = make_async_zerank_reranker(self._zerank, self.zerank_model)

    async def close_async(self) -> None:
        if self._async_client is not None:
            await self._async_client.close()
        # Close each reranker client if it exposes an async close — httpx-
        # based clients (Cohere, ZeroEntropy) and voyageai all do. Guard
        # individually so one failure doesn't leak the others.
        for name, client in (
            ("cohere", self._cohere),
            ("voyage", self._voyage),
            ("zerank", self._zerank),
        ):
            if client is None:
                continue
            close = getattr(client, "close", None)
            if close is None:
                continue
            try:
                result = close()
                if hasattr(result, "__await__"):
                    await result
            except Exception:
                pass

    async def run_async(self, query: str, tenant=None) -> list[ObjectID]:
        # Per-provider partial caching: each provider's score map can be
        # written independently, and on a re-run we only call providers
        # missing from the cache entry. So if Voyage fails once but Cohere
        # and Zerank succeed, the next run only re-invokes Voyage (avoiding
        # a re-bill on the two providers that worked the first time).
        PROVIDERS = ("cohere", "voyage", "zerank")
        SCORE_KEY = {p: f"{p}_scores" for p in PROVIDERS}

        entry = self.cache.queries.get(query)
        needed = (
            list(PROVIDERS) if entry is None
            else [p for p in PROVIDERS if SCORE_KEY[p] not in entry]
        )

        # Fully cached already — skip Weaviate and all rerankers.
        if entry is not None and not needed:
            return [ObjectID(object_id=d) for d in entry["hybrid_order"][:20]]

        # We need at least one reranker call, so we need doc_texts.
        # (We don't cache doc_texts to keep cache size bounded; one
        # Weaviate hybrid call per resume is much cheaper than the
        # rerank API calls it lets us skip.)
        from src.adapters.retrieval.weaviate_database import (
            async_weaviate_search_tool,
        )

        sources = await async_weaviate_search_tool(
            query=query,
            collection_name=self.collection_name,
            target_property_name=self.target_property,
            retrieved_k=self.retrieved_k,
            weaviate_async_client=self._async_client,
            return_vector=False,
            return_score=True,
        )
        doc_ids = [s.object_id for s in sources]
        doc_texts = [s.content for s in sources]
        n = len(sources)

        # Initialize entry on first sighting; persist hybrid_order
        # immediately so we don't redo the Weaviate call if every
        # reranker fails this turn.
        if entry is None:
            entry = {"hybrid_order": doc_ids}
            self.cache.queries[query] = entry

        # Schedule only the needed providers concurrently. Use
        # return_exceptions=True so one provider's failure doesn't
        # discard the other providers' successful results.
        provider_fns = {
            "cohere": self._cohere_fn,
            "voyage": self._voyage_fn,
            "zerank": self._zerank_fn,
        }
        tasks = {p: provider_fns[p](query, doc_texts, top_k=n) for p in needed}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        errors: dict[str, BaseException] = {}
        for p, result in zip(tasks.keys(), results):
            if isinstance(result, BaseException):
                errors[p] = result
                continue
            entry[SCORE_KEY[p]] = {
                doc_ids[item.index]: float(item.relevance_score)
                for item in result
            }

        # Persist whatever we got — including hybrid_order alone, in
        # the worst case where every reranker failed.
        async with self._save_lock:
            self.cache.save(self.cache_path)

        if errors:
            # Surface so the framework counts this query as failed; the
            # already-saved per-provider scores stay in the cache for the
            # next re-run to skip.
            details = "; ".join(f"{p}: {e!r}" for p, e in errors.items())
            raise RuntimeError(f"providers failed [{','.join(errors)}]: {details}")

        return [ObjectID(object_id=d) for d in entry["hybrid_order"][:20]]

    def run(self, query: str, tenant=None) -> list[ObjectID]:
        return asyncio.run(self.run_async(query, tenant))
