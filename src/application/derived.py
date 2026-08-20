"""DerivedSearchAgent: per-condition rankings entirely from a ScoreCache.

Zero API calls and no provider imports; ranking semantics live in
src.domain.fusion (the single implementation). Each derived condition specifies its
own `rerankers` subset so 2-way and 3-way conditions coexist over one cache.
"""
from __future__ import annotations

from query_agent_benchmarking import ObjectID

from src.adapters.cache import ScoreCache
from src.config import RRF_K
from src.domain.fusion import fused_ranking, rrf_fuse, rsf_fuse, singleton_ranking


class DerivedSearchAgent:
    """SearchAgent that produces a condition's ranking from cached scores.

    `condition` follows the same shape as src.conditions.Condition:
        - provider:        None (hybrid_only) | "cohere" | "voyage" | "zerank" | "hybrid"
        - fusion_method:   "rrf" | "rsf" (only used when provider == "hybrid")
        - weights:         {provider_name: float} for fusion (weights sum to 1)
        - rerankers:       tuple of provider names participating in fusion
                           (only meaningful when provider == "hybrid")
    """

    RRF_K = RRF_K
    SCORE_KEY = {
        "cohere": "cohere_scores",
        "voyage": "voyage_scores",
        "zerank": "zerank_scores",
    }

    def __init__(
        self,
        cache: ScoreCache,
        retrieved_k: int,
        condition,
        reranked_k: int = 20,
    ):
        self.cache = cache
        self.retrieved_k = retrieved_k
        self.condition = condition
        self.reranked_k = reranked_k

    async def initialize_async(self) -> None:
        pass

    async def close_async(self) -> None:
        pass

    def _scores_for(self, data: dict, provider: str, pool: list[str]) -> dict[str, float]:
        key = self.SCORE_KEY[provider]
        return {d: data[key][d] for d in pool if d in data[key]}

    def _derive(self, query: str) -> list[str]:
        data = self.cache.queries.get(query)
        if data is None:
            raise ValueError(
                f"Cache miss for query: {query[:80]!r}. Cache covers "
                f"{len(self.cache.queries)} queries."
            )

        hybrid_order: list[str] = data["hybrid_order"]
        pool = hybrid_order[: self.retrieved_k]

        # hybrid_only: return retrieved_k docs in hybrid order
        if self.condition.provider is None:
            return pool

        # Singleton conditions
        if self.condition.provider in self.SCORE_KEY:
            scores = self._scores_for(data, self.condition.provider, pool)
            return singleton_ranking(scores, self.reranked_k)

        # provider == "hybrid": fuse the subset specified by condition.rerankers
        rerankers = getattr(self.condition, "rerankers", None) or ("cohere", "voyage")
        weights = self.condition.weights or {r: 1.0 / len(rerankers) for r in rerankers}

        # Per-reranker scores filtered to the pool
        per_reranker = {r: self._scores_for(data, r, pool) for r in rerankers}

        if self.condition.fusion_method == "rsf":
            fused = rsf_fuse(per_reranker, pool, weights, rerankers)
        else:  # rrf (default)
            fused = rrf_fuse(per_reranker, weights, rerankers, self.RRF_K)

        return fused_ranking(fused, self.reranked_k)

    async def run_async(self, query: str, tenant=None) -> list[ObjectID]:
        return [ObjectID(object_id=d) for d in self._derive(query)]

    def run(self, query: str, tenant=None) -> list[ObjectID]:
        return [ObjectID(object_id=d) for d in self._derive(query)]
