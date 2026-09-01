# Architecture

`src/scaling_reranked_retrieval/` is a hexagonal (ports & adapters) layout.
The rule that makes the repo cheap to audit: **all experiment logic is pure
and offline; network access is confined to a handful of adapter/experiment
modules.**

## Layers

```
scaling_reranked_retrieval/
├── config.py       shared constants + the dataset/paths registry (DATASETS,
│                   get_results_dir) — importable from every layer
├── domain/         pure logic: no network, no filesystem, no provider SDKs
│   ├── fusion.py       RRF (η=60) / RSF fusion over per-provider score dicts,
│   │                   plus rank-list fusion for the listwise tier; RSF
│   │                   tie-break semantics are a pinned behavior contract
│   ├── conditions.py   the 12-condition equal-weight menu (build_menu)
│   ├── metrics.py      metric-name helpers + extra-recall-cutoff builder
│   └── aggregate.py    mean-across-queries / median-across-subsets
├── ports/          typing.Protocol seams — the only shapes the core sees
├── adapters/       infrastructure implementations
│   ├── cache.py        ScoreCache — resumable JSON score snapshots under
│   │                   results/<dataset>/caches/ (+ validate_cache_for_use)
│   ├── qab.py          bridge to the query-agent-benchmarking eval harness
│   └── retrieval/      Weaviate hybrid search + the three provider rerank
│                       callers (behind chunking + byte budgets)
└── application/    use cases
    ├── collect.py      CollectScoresAgent — LIVE score collection
    ├── derived.py      DerivedSearchAgent — cache-only derivation
    ├── experiments/    scripts that spend API calls
    └── analysis/       zero-network derivations over results/
```

`scripts/` holds one thin runnable wrapper per use case (`-h` on any of
them); no logic lives there.

## Ports

The application core talks to the outside world only through
`typing.Protocol` shapes in `ports/` — adapters satisfy them structurally, no
inheritance.

**Driving port** (how the eval harness invokes the core):

- `SearchAgent` — what `query_agent_benchmarking.run_search_eval` calls.
  Three implementations, one per mode:
  - `adapters.qab.RetrieverSearchAgent` — live retrieval + rerank
  - `application.collect.CollectScoresAgent` — live collection into the score cache
  - `application.derived.DerivedSearchAgent` — cache-only derivation (no network)

**Driven ports** (what the core needs from infrastructure):

- `RerankFn` — one provider's rerank call, as produced by the
  `make_*_reranker` factories in `adapters.retrieval.providers`
- `Retriever` — first-stage retrieval (`adapters.retrieval.base_retriever.BaseRetriever`,
  Weaviate hybrid search)
- `ScoreStore` — per-query score persistence (`adapters.cache.ScoreCache`)

## Dependency rules

- `domain/` may import `config` and other domain modules — never
  `application`, `adapters`, or any provider SDK. (`conditions.py` even
  redefines the `Provider`/`FusionMethod` literals locally rather than
  import them from the adapters layer.)
- Only `application.collect` may import the provider client constructors.
- `application/analysis/` is zero-network by construction: it reads
  `results/` and computes.

## The one-collection design

A cross-encoder scores each (query, document) pair independently of the rest
of the pool, so a single collection at k = 2,000 contains every score needed
at every shallower depth. The data flow is:

```
populate_db.py (once, LIVE)          Weaviate collection per subset
        │
run_experiment.py --collect-only     CollectScoresAgent fills
        │  (LIVE, resumable)         results/<subset>/caches/k2000.json
        ▼
k_sweep.py & analysis/ scripts       DerivedSearchAgent replays nested
   (offline, zero API calls)         prefixes of the cached pool through
                                     the condition menu → runs/, analysis
```

Because collection resumes per (provider, document) and holds a cached
query's `hybrid_order` fixed, the shipped pool caches pin the first stage:
re-collection reproduces the paper's candidate sets exactly even if the live
index has since drifted.
