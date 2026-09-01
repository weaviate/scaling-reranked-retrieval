"""Application layer — the experiment's use cases.

Orchestrates domain logic and adapters. Split by cost, same doctrine as the
run scripts in scripts/:

    experiments/  services that SPEND — live API calls or wall-clock
                  (collection, variance measurement, latency, listwise-LLM
                  reranking)
    analysis/     zero-network services — every result is re-derivable
                  offline from results/ artifacts in seconds-to-minutes

Shared services at this level:

    collect.py    CollectScoresAgent — one-time live score collection into
                  the ScoreCache (the only live-API module outside
                  experiments/)
    derived.py    DerivedSearchAgent — per-condition rankings derived from
                  the cache with zero API calls (the workhorse of every
                  derived sweep and analysis)
    queryset.py   gold-labelled query sets + cache validation over qab
                  datasets
    decompose.py  routing-vs-blending oracle decomposition over cached scores
    listwise.py   listwise-tier ranking store: load/validate cached LLM
                  rankings

Entry semantics live in scripts/ — every module here with a main() has a thin
wrapper there; nothing in this package parses sys.argv at import time.
"""
