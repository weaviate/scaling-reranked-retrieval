"""CollectScoresAgent: one-time real-call score collection at a chosen k.

The only live-API module in src/. Weaviate and provider clients are imported
lazily inside initialize_async so read-only consumers stay client-free.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from query_agent_benchmarking import ObjectID

from scaling_reranked_retrieval.adapters.cache import ScoreCache
from scaling_reranked_retrieval.config import EMBEDDING_MODEL

if TYPE_CHECKING:
    import weaviate


class CollectScoresAgent:
    """SearchAgent that records full cohere+voyage+zerank rerank scores per query.

    Returns the hybrid top-20 to satisfy the SearchAgent protocol; this run's
    metrics are not the experiment's reported numbers.
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
        self._zerank = None
        self._cohere_fn = None
        self._voyage_fn = None
        self._zerank_fn = None
        self._save_lock = asyncio.Lock()

    async def initialize_async(self) -> None:
        # Lazy imports keep provider clients out of module scope.
        import weaviate

        from scaling_reranked_retrieval.adapters.retrieval.clients import (
            get_cohere_async_client,
            get_voyage_async_client,
            get_zerank_async_client,
        )
        from scaling_reranked_retrieval.adapters.retrieval.providers import (
            make_async_cohere_reranker,
            make_async_voyage_reranker,
            make_async_zerank_reranker,
        )
        from scaling_reranked_retrieval.adapters.retrieval.embeddings_registry import (
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
        # Chunking-aware wrappers handle retrieved_k above the 1000-doc per-call API limit.
        self._cohere_fn = make_async_cohere_reranker(self._cohere, self.cohere_model)
        self._voyage_fn = make_async_voyage_reranker(self._voyage, self.voyage_model)
        self._zerank_fn = make_async_zerank_reranker(self._zerank, self.zerank_model)

    async def close_async(self) -> None:
        if self._async_client is not None:
            await self._async_client.close()
        # Close each client that exposes close(); guard individually so one failure doesn't leak the others.
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
        # Per-(provider, doc) resume: only score pool docs missing from each
        # provider's map, then merge — exact because providers score pairs
        # independently of batch composition.
        PROVIDERS = ("cohere", "voyage", "zerank")
        SCORE_KEY = {p: f"{p}_scores" for p in PROVIDERS}

        entry = self.cache.queries.get(query)

        def missing_docs(pool: list[str]) -> dict[str, list[str]]:
            return {
                p: [d for d in pool if d not in entry.get(SCORE_KEY[p], {})]
                for p in PROVIDERS
            }

        # Fully covered — skip everything. Coverage is per doc, so refreshed pools aren't silently skipped.
        if entry is not None and not any(missing_docs(entry["hybrid_order"]).values()):
            return [ObjectID(object_id=d) for d in entry["hybrid_order"][:20]]

        # New query: one hybrid retrieval establishes hybrid_order and texts.
        # Resume: hybrid_order is held fixed by the cache (no retrieval query);
        # missing texts are fetched by dataset_id.
        from scaling_reranked_retrieval.adapters.retrieval.weaviate_database import (
            async_fetch_texts_by_id,
            async_weaviate_search_tool,
        )

        if entry is None:
            sources = await async_weaviate_search_tool(
                query=query,
                collection_name=self.collection_name,
                target_property_name=self.target_property,
                retrieved_k=self.retrieved_k,
                weaviate_async_client=self._async_client,
                return_vector=False,
                return_score=True,
            )
            text_by_id = {s.object_id: s.content for s in sources}
            # Persist hybrid_order immediately in case every reranker fails this turn.
            entry = {"hybrid_order": [s.object_id for s in sources]}
            self.cache.queries[query] = entry
            missing = missing_docs(entry["hybrid_order"])
        else:
            missing = missing_docs(entry["hybrid_order"])
            union = list(dict.fromkeys(d for docs in missing.values() for d in docs))
            text_by_id = await async_fetch_texts_by_id(
                union,
                self.collection_name,
                self.target_property,
                self._async_client,
            )

        # Ids the fetch couldn't resolve are skipped, not fatal; derived runs tolerate score gaps.
        needed = {
            p: [d for d in docs if d in text_by_id]
            for p, docs in missing.items()
        }
        needed = {p: docs for p, docs in needed.items() if docs}

        # return_exceptions=True so one provider's failure doesn't discard the others' results.
        provider_fns = {
            "cohere": self._cohere_fn,
            "voyage": self._voyage_fn,
            "zerank": self._zerank_fn,
        }
        tasks = {
            p: provider_fns[p](query, [text_by_id[d] for d in docs], top_k=len(docs))
            for p, docs in needed.items()
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        errors: dict[str, BaseException] = {}
        for (p, docs), result in zip(needed.items(), results):
            if isinstance(result, BaseException):
                errors[p] = result
                continue
            entry.setdefault(SCORE_KEY[p], {}).update(
                {docs[item.index]: float(item.relevance_score) for item in result}
            )

        # Persist whatever we got — hybrid_order alone in the worst case.
        async with self._save_lock:
            self.cache.save(self.cache_path)

        if errors:
            # Raise so the framework counts the query failed; saved scores let re-runs skip.
            details = "; ".join(f"{p}: {e!r}" for p, e in errors.items())
            raise RuntimeError(f"providers failed [{','.join(errors)}]: {details}")

        return [ObjectID(object_id=d) for d in entry["hybrid_order"][:20]]

    def run(self, query: str, tenant=None) -> list[ObjectID]:
        return asyncio.run(self.run_async(query, tenant))
