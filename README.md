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
further API calls. That is the economics of this whole repo — every analysis
is a fast offline derivation over one expensive, resumable collection pass
per dataset.

**What is distributed.** The first-stage hybrid retrieval pools ship with the
repo (`results/bright_<subset>/caches/k2000.json` — per-query document
rankings, no reranker scores). Collection resumes into these pinned pools, so
your score collection reproduces the paper's exact candidate sets. Reranker
scores, run summaries, and analysis artifacts are not distributed; rebuilding
them requires API keys for the providers being measured.

This repository is a research artifact accompanying the paper — organized as
a library for readability, provided as-is, and not maintained as an evolving
package.

## Quickstart

```bash
uv sync

export COHERE_API_KEY=... VOYAGE_API_KEY=... ZERANK_API_KEY=...

# Fill the shipped k=2000 pool with cross-encoder scores (LIVE, resumable):
uv run python scripts/run_experiment.py --dataset biology --retrieved-k 2000 --collect-only

# Derive the entire depth sweep offline (zero API calls):
uv run python scripts/k_sweep.py biology
```

Only the live-call scripts need keys; every analysis runs offline from
`results/`. See [docs/getting-started.md](docs/getting-started.md) for the
full setup, including the Weaviate and OpenAI stages.

## Layout

The `scaling_reranked_retrieval` package (under `src/`) is hexagonal
(ports & adapters); `scripts/` holds the run scripts — each a thin wrapper
over one application use case (`-h` on any of them).

```
src/scaling_reranked_retrieval/
  domain/        fusion math (RRF η=60 / RSF), the condition menu, metrics
  ports/         Protocol seams (SearchAgent, RerankFn, Retriever, ScoreStore)
  adapters/      score cache, eval-harness bridge, and the retrieval layer
                 (Weaviate hybrid + provider rerank callers)
  application/   use cases: experiments/ spend API calls; analysis/ is
                 zero-network over results/
scripts/         one run script per use case
docs/            architecture, data formats, and the reproduction runbook
```

## Documentation

- [docs/getting-started.md](docs/getting-started.md) — install, API keys,
  the core collect-then-derive loop, tests
- [docs/reproducing.md](docs/reproducing.md) — every paper section mapped to
  its command, plus guidance on reading the numbers
- [docs/architecture.md](docs/architecture.md) — the hexagon: layers, ports,
  dependency rules, and the one-collection data flow
- [docs/data-formats.md](docs/data-formats.md) — schemas for the shipped
  pool caches and every generated artifact
