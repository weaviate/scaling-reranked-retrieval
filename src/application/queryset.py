"""Query universe + drop accounting over a score cache.

Builds the "all-three-present intersection" every cross-reranker analysis
runs on, and the standard load-and-validate entry (cache + query set) for a
dataset slug. Zero provider imports; qab is used only to load gold sets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from query_agent_benchmarking.internal.adapters.dataset import (
    in_memory_dataset_loader,
)

from src.adapters.cache import ScoreCache, validate_cache_for_use
from src.config import (
    CACHE_K,
    DATASETS,
    MODEL_OVERRIDES,
    PROVIDERS,
    get_results_dir,
)


@dataclass
class QuerySet:
    # query text -> gold doc-id set, over the all-three-present intersection.
    gold: dict[str, set[str]] = field(default_factory=dict)
    query_ids: dict[str, str] = field(default_factory=dict)  # text -> qab query_id
    drops: dict[str, list[str]] = field(default_factory=dict)  # provider -> [qid]
    # Per provider: that provider's own present-query set (for regression).
    present_by_provider: dict[str, dict[str, set[str]]] = field(default_factory=dict)
    no_gold: list[str] = field(default_factory=list)


def build_query_set(cache: ScoreCache, dataset_name: str) -> QuerySet:
    """Map cache queries to gold sets and compute the analysis intersection.

    The intersection is the set of queries scored by ALL THREE providers.
    Provider presence is k-independent: if a provider scored a query at all
    it scored the full retrieved_k=2000 pool, so a doc present at k=2000 is
    present at every smaller k.
    """
    _, queries = in_memory_dataset_loader(dataset_name, queries_only=True)
    gold_by_text: dict[str, set[str]] = {}
    qid_by_text: dict[str, str] = {}
    for q in queries:
        gold_by_text[q.question] = {str(d) for d in q.dataset_ids}
        qid_by_text[q.question] = q.query_id or q.question[:64]

    qs = QuerySet()
    qs.present_by_provider = {p: {} for p in PROVIDERS}
    drops: dict[str, list[str]] = {p: [] for p in PROVIDERS}

    for text, entry in cache.queries.items():
        gold = gold_by_text.get(text)
        if gold is None:
            qs.no_gold.append(text[:64])
            continue
        qid = qid_by_text.get(text, text[:64])
        present = [p for p in PROVIDERS if entry.get(f"{p}_scores")]
        for p in present:
            qs.present_by_provider[p][text] = gold
        for p in PROVIDERS:
            if p not in present:
                drops[p].append(qid)
        if len(present) == len(PROVIDERS):
            qs.gold[text] = gold
            qs.query_ids[text] = qid

    qs.drops = {p: ids for p, ids in drops.items() if ids}
    return qs


def load_and_validate(dataset_slug: str) -> Optional[tuple[ScoreCache, QuerySet]]:
    """Load + validate the k=2000 cache and build the query set, or None."""
    _cfg = DATASETS[dataset_slug]
    dataset_name, collection_name = _cfg.qab_name, _cfg.collection
    cache_path = get_results_dir(dataset_slug) / "caches" / f"k{CACHE_K}.json"
    if not cache_path.exists():
        print(f"  [skip] no cache at {cache_path}")
        return None
    cache = ScoreCache.load(cache_path)
    validate_cache_for_use(
        cache,
        needed_retrieved_k=CACHE_K,
        expected_model_overrides=MODEL_OVERRIDES,
        expected_dataset=dataset_name,
        expected_collection=collection_name,
    )
    qs = build_query_set(cache, dataset_name)
    print(
        f"  cache: {len(cache.queries)} queries, intersection "
        f"{len(qs.gold)}, drops {qs.drops or '{}'}"
    )
    return cache, qs
