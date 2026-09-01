"""Adapters — implementations of the driven ports over real infrastructure.

    cache.py      ScoreStore adapter: resumable per-query rerank-score
                  snapshots as JSON under results/<dataset>/caches/.
    qab.py        adapter for the query_agent_benchmarking eval harness:
                  the SearchAgent bridge plus the qab runtime guards
                  (version check + corpus-loader memoization via setup()).
    retrieval/    retrieval infrastructure: Weaviate hybrid search, provider
                  rerank callers (Cohere / Voyage / ZeroEntropy) with
                  chunking and byte-budget handling, and the RRF/RSF
                  implementations. See its __init__ for the module map.

Import discipline: modules that instantiate provider SDK clients
(retrieval/clients.py, retrieval/providers.py) are imported only by the
live-call application services (collect, latency, score-variance) — the
zero-network analysis paths never touch them.
"""
