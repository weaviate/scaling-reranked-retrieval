"""mixture-of-rerankers library — hexagonal layout.

    domain/       pure experiment logic (fusion math, condition menu,
                  metrics, aggregation) — no I/O, no SDKs
    ports/        Protocol interfaces between the core and the world
                  (SearchAgent, RerankFn, Retriever, ScoreStore)
    adapters/     infrastructure implementing the ports: the score cache,
                  the qab SearchAgent bridge, and the retrieval layer
                  (Weaviate hybrid search + provider rerank callers)
    application/  use cases orchestrating domain + adapters, split by cost:
                  experiments/ spend API calls; analysis/ is zero-network

Shared kernel at this level: config.py (paths, dataset registry, constants).

Entry points live in scripts/ — thin wrappers over application main()s.
Dependency rule: domain ← application → adapters; scripts import only
application (and src root). Nothing in src imports from scripts.

Import hygiene contract: importing the domain layer or any zero-network
application path must not import a reranker provider client (cohere,
voyageai, zeroentropy) — the offline analyses depend on it. Only
application.collect and the live experiment services may touch
adapters.retrieval.clients / providers. (weaviate unavoidably rides along
with query_agent_benchmarking's own module-scope imports; no client is
instantiated.)
"""
