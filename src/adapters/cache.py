"""Persistent per-query rerank-score cache.

The idea: run real Weaviate + Cohere + Voyage + Zerank-2 calls once at a large
retrieved_k and cache every (query, doc_id) → (cohere_score, voyage_score,
zerank_score). Then derive results for any smaller retrieved_k by filtering
the cache to the top-N hybrid pool and recomputing rankings/normalizations
locally — no API calls and no model inference.

This is exact (not approximate) for all three rerankers because they are all
cross-encoders that score each (query, doc) pair independently of batch
composition.

For RSF specifically: min-max normalization is pool-dependent, so it must be
recomputed at derive time over the restricted pool — not reused from the
collection-time normalization. The raw scores are pool-independent; the
normalization is not.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ScoreCache:
    """Persistent per-query cache of hybrid order + per-provider scores.

    Layout:
        metadata: dict      # dataset, collection, retrieved_k, model overrides
        queries:  dict[str, {hybrid_order: [doc_id...],
                             cohere_scores: {doc_id: float},
                             voyage_scores: {doc_id: float},
                             zerank_scores: {doc_id: float}}]
    """

    metadata: dict = field(default_factory=dict)
    queries: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "ScoreCache":
        with open(path) as f:
            d = json.load(f)
        return cls(metadata=d.get("metadata", {}), queries=d.get("queries", {}))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump({"metadata": self.metadata, "queries": self.queries}, f)
        tmp.replace(path)


def validate_cache_for_use(
    cache: ScoreCache,
    needed_retrieved_k: int,
    expected_model_overrides: dict,
    expected_dataset: str,
    expected_collection: str,
) -> None:
    """Raise if the cache is not suitable for a derived run at the given k."""
    meta = cache.metadata
    cached_k = meta.get("retrieved_k")
    if cached_k is None or cached_k < needed_retrieved_k:
        raise ValueError(
            f"Cache was collected at retrieved_k={cached_k}, but derived run "
            f"needs retrieved_k={needed_retrieved_k}. Recollect at a larger k."
        )
    if meta.get("model_overrides") != expected_model_overrides:
        raise ValueError(
            f"Cache model overrides {meta.get('model_overrides')} do not match "
            f"expected {expected_model_overrides}. Cache is stale; recollect."
        )
    if meta.get("dataset") != expected_dataset:
        raise ValueError(
            f"Cache dataset {meta.get('dataset')!r} != expected {expected_dataset!r}."
        )
    if meta.get("collection") != expected_collection:
        raise ValueError(
            f"Cache collection {meta.get('collection')!r} != expected "
            f"{expected_collection!r}."
        )
