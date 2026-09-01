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
        # Per-(provider, doc) partial caching: on a re-run we only score the
        # pool docs missing from each provider's score map, and merge the new
        # scores in. Exact because all three providers are cross-encoders
        # that score each (query, doc) pair independently of batch
        # composition (see adapters/cache.py). This covers both resume cases:
        # a provider that failed entirely on a previous run, and a refreshed
        # hybrid_order containing docs the old score maps never saw.
        PROVIDERS = ("cohere", "voyage", "zerank")
        SCORE_KEY = {p: f"{p}_scores" for p in PROVIDERS}

        entry = self.cache.queries.get(query)

        def missing_docs(pool: list[str]) -> dict[str, list[str]]:
            return {
                p: [d for d in pool if d not in entry.get(SCORE_KEY[p], {})]
                for p in PROVIDERS
            }

        # Fully covered already — skip Weaviate and all rerankers. Coverage
        # is per doc, not just key presence, so entries whose pool was
        # refreshed after collection are not silently skipped.
        if entry is not None and not any(missing_docs(entry["hybrid_order"]).values()):
            return [ObjectID(object_id=d) for d in entry["hybrid_order"][:20]]

        # We need at least one reranker call, so we need doc texts (not
        # cached, to keep cache size bounded). Two cases:
        #
        # - New query: one hybrid retrieval establishes hybrid_order AND
        #   supplies every text.
        # - Resume of an existing entry: hybrid_order is fixed by the cache
        #   (it is the experiment's pool; derived runs slice prefixes of
        #   it), so no retrieval query runs at all — the missing docs'
        #   texts are fetched by dataset_id. Collection therefore has zero
        #   dependence on retrieval reproducibility.
        from src.adapters.retrieval.weaviate_database import (
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
            # Persist hybrid_order immediately so we don't redo the
            # Weaviate call if every reranker fails this turn.
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

        # Score only the pool docs that lack a score AND have a text. An id
        # the fetch couldn't resolve (e.g. a doc the collection is missing
        # relative to the corpus) is skipped, not fatal; derived runs
        # already tolerate score gaps by dropping the doc from that
        # provider's ranking.
        needed = {
            p: [d for d in docs if d in text_by_id]
            for p, docs in missing.items()
        }
        needed = {p: docs for p, docs in needed.items() if docs}

        # Schedule only the needed providers concurrently. Use
        # return_exceptions=True so one provider's failure doesn't
        # discard the other providers' successful results.
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
