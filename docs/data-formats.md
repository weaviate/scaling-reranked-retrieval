# Data formats

Everything lives under `results/bright_<subset>/` for the five BRIGHT
subsets (`biology`, `earth_science`, `economics`, `psychology`,
`robotics`), plus `results/listwise/` for the listwise tier.

## Score caches: `caches/k2000.json`

The one file everything else derives from. **The shipped copies contain the
first-stage pools only. Collection adds the score maps.**

```jsonc
{
  "metadata": {
    "dataset": "bright/biology",          // qab dataset id
    "collection": "BrightBiology_Default", // Weaviate collection name
    "retrieved_k": 2000,                   // pool depth
    "model_overrides": { /* embedding/search config */ }
  },
  "queries": {
    "<query_id>": {
      // First-stage hybrid ranking, best-first, exactly retrieved_k ids.
      // SHIPPED. Held fixed on resume: collection never re-retrieves it.
      "hybrid_order": ["doc/a.txt", "doc/b.txt", ...],

      // Added by run_experiment.py --collect-only (doc id -> score).
      // NOT shipped. Keys are a subset of hybrid_order, resumable
      // per (provider, doc).
      "cohere_scores": {"doc/a.txt": 0.919, ...},
      "voyage_scores": {"doc/a.txt": 0.734, ...},
      "zerank_scores": {"doc/a.txt": 0.839, ...}
    }
  }
}
```

Shipped query counts: biology 103, earth_science 116, economics 103,
psychology 101, robotics 101.

`validate_cache_for_use` (in `scaling_reranked_retrieval.adapters.cache`)
gates every derived run: the cache must have been collected at
`retrieved_k >= ` the requested depth with matching dataset / collection /
model overrides.

## Derived runs: `runs/k{N}_from_k{M}.json`

One file per depth, written by `scripts/k_sweep.py` /
`scripts/run_experiment.py --from-cache`: per-condition metrics (nDCG@10,
Recall@k, Success@1) for the depth-`N` prefix of the depth-`M` cache,
evaluated over the 12-condition equal-weight menu. `runs_rk{R}/` variants
(e.g. `runs_rk100/`) retain `R` candidates from the cross-encoder instead of
the default 20, for the deep-recall analyses.

## Listwise tier: `results/listwise/`

Built by `scripts/listwise_rerank.py` and consumed by the `listwise_*`
analysis scripts:

- `pools/`: the frozen top-20 inputs per (subset, condition), derived
  offline from the score caches
- `cache/`: one JSONL per (subset, model, effort, condition, pool) with
  every (query, trial) ranking, e.g.
  `biology__gpt-5.6-terra__none__zerank-only__first2000__top20.jsonl`.
  Cached rankings are never re-bought. Re-runs only pay for missing
  (query, trial) calls.
- `fusion/`, `oracle_routing/`, `self_oracle/`, `unique_successes/`:
  analysis outputs over those caches

`results/raw/listwise_top100/` holds the n=100 extended-window variant
(`scripts/listwise_top100.py`), same shape.

## Provenance and drift

The shipped pools were retrieved with Weaviate hybrid search (BM25 +
Snowflake Arctic 2.0 embeddings, relative-score fusion, α = 0.75). Live
re-retrieval against a freshly populated collection can differ slightly from
these pools (server-side first-stage drift). All paper numbers are computed
against the shipped, pinned pools, which is why collection resumes into them
rather than re-retrieving.
