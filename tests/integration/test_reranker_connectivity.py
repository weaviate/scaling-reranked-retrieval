"""Live connectivity spot-checks for the three reranker providers.

One tiny rerank call per provider (1 query, 2 docs) through the same
client + adapter path CollectScoresAgent uses, with the paper's model
overrides — so a pass means the exact (provider, model) pair the collection
run will bill against is reachable with the keys in this shell.

Each test SKIPS (not fails) when its API key env var is unset, so the suite
is safe to run in any shell:

    uv run pytest tests/integration -v
"""
import math
import os

import pytest

from scaling_reranked_retrieval.config import MODEL_OVERRIDES

pytestmark = pytest.mark.integration

QUERY = "What organelle produces most of the cell's ATP?"
DOCS = [
    "The mitochondrion generates most of the cell's supply of ATP.",
    "ROS 2 nodes communicate with each other over topics and services.",
]


def _check(items):
    assert len(items) == 2
    scores = {i.index: i.relevance_score for i in items}
    assert set(scores) == {0, 1}
    assert all(isinstance(s, float) and math.isfinite(s) for s in scores.values())
    # Not a model-quality test, but with a pair this contrastive an inverted
    # ranking means the wrong model/endpoint answered, not a close call.
    assert scores[0] > scores[1]


@pytest.mark.skipif(not os.getenv("COHERE_API_KEY"), reason="COHERE_API_KEY not set")
def test_cohere_reachable():
    from scaling_reranked_retrieval.adapters.retrieval.clients import get_cohere_client
    from scaling_reranked_retrieval.adapters.retrieval.providers import make_cohere_reranker

    fn = make_cohere_reranker(get_cohere_client().client, MODEL_OVERRIDES["cohere"])
    _check(fn(QUERY, DOCS, top_k=2))


@pytest.mark.skipif(not os.getenv("VOYAGE_API_KEY"), reason="VOYAGE_API_KEY not set")
def test_voyage_reachable():
    from scaling_reranked_retrieval.adapters.retrieval.clients import get_voyage_client
    from scaling_reranked_retrieval.adapters.retrieval.providers import make_voyage_reranker

    fn = make_voyage_reranker(get_voyage_client().client, MODEL_OVERRIDES["voyage"])
    _check(fn(QUERY, DOCS, top_k=2))


@pytest.mark.skipif(not os.getenv("ZERANK_API_KEY"), reason="ZERANK_API_KEY not set")
def test_zerank_reachable():
    from scaling_reranked_retrieval.adapters.retrieval.clients import get_zerank_client
    from scaling_reranked_retrieval.adapters.retrieval.providers import make_zerank_reranker

    fn = make_zerank_reranker(get_zerank_client().client, MODEL_OVERRIDES["zerank"])
    _check(fn(QUERY, DOCS, top_k=2))
