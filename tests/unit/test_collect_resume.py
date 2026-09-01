"""Per-(provider, doc) resume semantics of CollectScoresAgent.run_async.

Pins the behavior the refreshed earth_science/robotics caches rely on: an
entry whose hybrid_order was replaced with a fresh pool, but whose score maps
were carried over from the old collection, must trigger scoring for exactly
the missing (provider, doc) pairs — merged, not overwritten. The pool is
fixed by the cache: resume must fetch missing texts by dataset_id and never
run a retrieval query. No network: Weaviate and the provider fns are stubbed.
"""
import asyncio
from pathlib import Path

from src.adapters.cache import ScoreCache
from src.application.collect import CollectScoresAgent
import src.adapters.retrieval.weaviate_database as wdb


class _Source:
    def __init__(self, object_id, content):
        self.object_id = object_id
        self.content = content


class _Item:
    def __init__(self, index, relevance_score):
        self.index = index
        self.relevance_score = relevance_score


def _no_search(monkeypatch):
    def boom(**kwargs):
        raise AssertionError("retrieval query must not run for an existing entry")
    monkeypatch.setattr(wdb, "async_weaviate_search_tool", boom)


def _fetch_from(corpus):
    async def fetch(doc_ids, collection_name, target_property, client):
        return {d: f"text of {d}" for d in doc_ids if d in corpus}
    return fetch


def _make_agent(tmp_path: Path, entry, calls: dict):
    queries = {"q": dict(entry)} if entry is not None else {}
    cache = ScoreCache(metadata={}, queries=queries)

    def make_fn(provider):
        async def fn(query, documents, top_k):
            calls[provider] = list(documents)
            return [_Item(i, 0.5) for i in range(len(documents))]
        return fn

    agent = CollectScoresAgent(
        collection_name="C", target_property="content", retrieved_k=2000,
        cache=cache, cache_path=tmp_path / "k.json",
        cohere_model="m", voyage_model="m", zerank_model="m",
    )
    agent._cohere_fn = make_fn("cohere")
    agent._voyage_fn = make_fn("voyage")
    agent._zerank_fn = make_fn("zerank")
    return agent, cache


def test_refreshed_pool_scores_only_missing_pairs(tmp_path, monkeypatch):
    entry = {
        "hybrid_order": ["a", "b", "c"],   # refreshed pool
        "cohere_scores": {"a": 0.9, "b": 0.8},  # c missing
        "voyage_scores": {"a": 0.9, "b": 0.8, "c": 0.7},  # complete
        "zerank_scores": {},               # all missing
    }
    calls = {}
    agent, cache = _make_agent(tmp_path, entry, calls)
    _no_search(monkeypatch)
    monkeypatch.setattr(wdb, "async_fetch_texts_by_id", _fetch_from({"a", "b", "c"}))
    asyncio.run(agent.run_async("q"))

    assert calls == {
        "cohere": ["text of c"],
        "zerank": ["text of a", "text of b", "text of c"],
    }  # voyage complete -> not called
    e = cache.queries["q"]
    assert e["hybrid_order"] == ["a", "b", "c"]  # pool untouched
    assert e["cohere_scores"] == {"a": 0.9, "b": 0.8, "c": 0.5}  # merged
    assert e["voyage_scores"]["c"] == 0.7  # untouched
    assert set(e["zerank_scores"]) == {"a", "b", "c"}


def test_fully_covered_entry_skips_everything(tmp_path, monkeypatch):
    entry = {
        "hybrid_order": ["a", "b"],
        "cohere_scores": {"a": 1.0, "b": 1.0},
        "voyage_scores": {"a": 1.0, "b": 1.0},
        "zerank_scores": {"a": 1.0, "b": 1.0},
    }
    calls = {}
    agent, _ = _make_agent(tmp_path, entry, calls)
    _no_search(monkeypatch)

    def boom(*args, **kwargs):
        raise AssertionError("no fetch should run for a covered entry")
    monkeypatch.setattr(wdb, "async_fetch_texts_by_id", boom)

    result = asyncio.run(agent.run_async("q"))
    assert calls == {}
    assert [o.object_id for o in result] == ["a", "b"]


def test_unfetchable_missing_doc_is_skipped_not_fatal(tmp_path, monkeypatch):
    # 'c' lacks a cohere score but doesn't exist in the collection (corpus
    # shortfall): no text, so it must be skipped without raising.
    entry = {
        "hybrid_order": ["a", "b", "c"],
        "cohere_scores": {"a": 1.0, "b": 1.0},
        "voyage_scores": {"a": 1.0, "b": 1.0, "c": 1.0},
        "zerank_scores": {"a": 1.0, "b": 1.0, "c": 1.0},
    }
    calls = {}
    agent, cache = _make_agent(tmp_path, entry, calls)
    _no_search(monkeypatch)
    monkeypatch.setattr(wdb, "async_fetch_texts_by_id", _fetch_from({"a", "b"}))
    result = asyncio.run(agent.run_async("q"))
    assert calls == {}
    assert "c" not in cache.queries["q"]["cohere_scores"]
    assert [o.object_id for o in result] == ["a", "b", "c"]


def test_new_query_uses_retrieval_and_scores_full_pool(tmp_path, monkeypatch):
    calls = {}
    agent, cache = _make_agent(tmp_path, entry=None, calls=calls)

    async def search(**kwargs):
        return [_Source(d, f"text of {d}") for d in ["a", "b"]]
    monkeypatch.setattr(wdb, "async_weaviate_search_tool", search)

    def boom(*args, **kwargs):
        raise AssertionError("fetch-by-id is for resume, not first sighting")
    monkeypatch.setattr(wdb, "async_fetch_texts_by_id", boom)

    asyncio.run(agent.run_async("q"))
    e = cache.queries["q"]
    assert e["hybrid_order"] == ["a", "b"]
    assert calls == {p: ["text of a", "text of b"]
                     for p in ("cohere", "voyage", "zerank")}
    assert all(set(e[f"{p}_scores"]) == {"a", "b"}
               for p in ("cohere", "voyage", "zerank"))
