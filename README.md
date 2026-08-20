# Scaling Reranked Retrieval

Companion code for **_Scaling Reranked Retrieval_**, a study of where the next
unit of inference compute is best spent in a staged retrieval pipeline:
**depth** (deeper candidate pools), **stages** (adding a cross-encoder, then a
listwise pass), or **width** (running several models at one stage and fusing
their judgments).

The measured pipeline, over five reasoning-intensive BRIGHT subsets
(Biology, Earth Science, Economics, Psychology, Robotics — ~100 queries each):

1. **First stage** — Weaviate hybrid search (BM25 + Snowflake Arctic 2.0
   embeddings, relative-score fusion, α = 0.75) retrieves a pool of
   k ∈ {100 … 2,000} candidates.
2. **Cross-encoder stage** — one of three commercial rerankers rescores the
   pool: Cohere `rerank-v4.0-pro`, Voyage `rerank-2.5`, ZeroEntropy
   `zerank-2` — or an equal-weight RRF/RSF fusion of them.
3. **Listwise stage** — an LLM (GPT-5.4 Mini, GPT-5.6 Luna, or GPT-5.6 Terra,
   reasoning effort `none`, 3 trials/query) jointly reorders the retained
   top-20.

Ordering quality is reported as **nDCG@10**, retention as **Recall@k**, and
head-of-list precision as **Success@1**.

**The one-collection design.** A cross-encoder scores each (query, document)
pair independently of the rest of the pool, so a single collection at
k = 2,000 already contains every score needed at every shallower depth: the
entire depth sweep is derived as nested prefixes of one retrieval, with zero
further API calls. That is the economics of this whole repo — every
`scripts/` analysis below is a fast offline derivation over one expensive,
resumable collection pass per dataset.

**Results are not distributed.** Everything under `results/` (score caches,
run summaries, analysis artifacts) is reproduced by the runbook below;
collection requires API keys for the providers being measured.

This repository is a research artifact accompanying the paper — organized as
a library for readability, provided as-is, and not maintained as an evolving
package.

## Layout

`src/` is hexagonal (ports & adapters); `scripts/` holds the run scripts —
each a thin wrapper over one application use case (`-h` on any of them).

```
src/domain/        fusion math (RRF η=60 / RSF), the condition menu, metrics
src/ports/         Protocol seams (SearchAgent, RerankFn, Retriever, ScoreStore)
src/adapters/      score cache, eval-harness bridge, and the retrieval layer
                   (Weaviate hybrid + provider rerank callers)
src/application/   use cases: experiments/ spend API calls; analysis/ is
                   zero-network over results/
scripts/           one run script per use case
```

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for dependency rules and the
behavior-preservation contract.

## Setup

```bash
uv sync

export WEAVIATE_URL=... WEAVIATE_API_KEY=...          # first stage
export COHERE_API_KEY=... VOYAGE_API_KEY=... ZERANK_API_KEY=...  # CE stage
export OPENAI_API_KEY=...                              # listwise stage
```

Only the live-call scripts need keys; every analysis runs offline from
`results/`. Always invoke via `uv run` (syncs the environment to `uv.lock`).

## Reproducing the paper

Each command maps to the section of the paper it produces data for.

### The depth sweep (§ Scaling Retrieval, § Singleton Cross-Encoders)

```bash
# Collect once per dataset at k=2000 (LIVE; resumable — re-runs skip cached queries):
uv run python scripts/run_experiment.py --dataset biology --retrieved-k 2000 --collect-only

# Derive the full nested-prefix depth sweep (zero API calls):
uv run python scripts/k_sweep.py biology
```

Datasets: `biology`, `earth_science`, `economics`, `psychology`, `robotics`.
Derived runs land in `results/<subset>/runs/k{N}_from_k2000.json`.

### Native-limit retrieval stability (Appendix: Retrieval Stability)

The nested-prefix sweep holds the initial ranking fixed; the robustness check
reruns native retrieval five times at each limit:

```bash
uv run python scripts/hybrid_variance.py --n-trials 5   # LIVE Weaviate calls
uv run python scripts/score_variance.py                 # reranker determinism
```

### Reranked depth and the capture ceiling (§ Reranked Depth)

Retaining 100 instead of 20 candidates from the cross-encoder, to measure
Recall@50/@100 against the first-stage ceiling:

```bash
uv run python scripts/k_sweep.py biology --reranked-k 100   # -> runs_rk100/
uv run python scripts/success_at_20.py
uv run python scripts/singleton_deep_recall.py
```

### The listwise stage (§ Singleton Listwise Rerankers, § Extended Window)

```bash
uv run python scripts/listwise_rerank.py --build-all-pools   # offline pool build
uv run python scripts/listwise_rerank.py --all-domains --model <model> --dry-run  # cost preflight, $0
uv run python scripts/listwise_rerank.py --all-domains --model <model>            # LIVE, 3 trials/query
uv run python scripts/listwise_top100.py -h                  # the n=100 window extension
```

Every (query, trial) ranking is cached and resumable — re-runs only pay for
missing calls. Models without a `MODEL_PRICES` entry require explicit
`--price-in/--price-out` ($/1M tokens).

### Width: fusion and disagreement (§ Exploring Width in Reranking)

```bash
uv run python scripts/equal_weight.py               # equal-weight fusion vs singletons (CE stage)
uv run python scripts/unique_successes.py           # per-model unique rank-1 successes (CE stage)
uv run python scripts/agreement.py --all-datasets   # rank-1 agreement / decorrelation
uv run python scripts/listwise_fusion.py            # equal-weight RRF at the listwise stage
uv run python scripts/listwise_unique_successes.py  # unique successes, listwise tier
```

### Routing headroom and its null (§ Discussion — query-dependent routing)

The oracle ceilings behind the routing-as-future-work claim, and the
winner's-curse controls that keep them honest:

```bash
uv run python scripts/oracle_config.py --k 2000        # routing vs blending decomposition
uv run python scripts/noise_null.py --singleton-only   # selection-on-noise null (CE)
uv run python scripts/listwise_oracle_routing.py       # selection ceiling, listwise tier
uv run python scripts/listwise_self_oracle.py          # self-ensemble control, listwise tier
```

### Deployment cost (§ The Cost of Depth, Width, and Stages)

```bash
uv run python scripts/latency_measurement.py --dataset biology   # LIVE timing calls
```

## Notes

- **Fusion menu is equal-weight only** (`src.domain.conditions.CONDITIONS`):
  baselines, the three singletons, and equal-weight pair/3-way RRF+RSF blends.
  The paper's width result is about untuned, training-free fusion; no tilted
  weights exist in the code.
- **RSF ties.** RSF fusion produces exact score ties whose break order is
  `PYTHONHASHSEED`-dependent (~1 query of wobble); `scripts/noise_null.py`
  pins the seed for its bit-reproducible sweep. Differences below ~0.01 on a
  single RSF cell are noise.
- With ~100 queries per subset, 0.01 on a hit-based metric ≈ one query on a
  subset (≈ one query per subset for a cross-subset mean) — treat sub-0.02
  deltas accordingly.
