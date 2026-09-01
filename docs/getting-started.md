# Getting started

## Install

```bash
uv sync
```

Always invoke via `uv run` — it keeps the environment synced to `uv.lock`.
The project installs as the `scaling_reranked_retrieval` package (editable),
so both `scripts/` and your own code can do:

```python
from scaling_reranked_retrieval.domain.fusion import rrf_fuse, rsf_fuse
from scaling_reranked_retrieval.domain.conditions import build_menu
```

## What ships, and what you rebuild

The repository distributes the **first-stage hybrid retrieval pools** — one
cache per BRIGHT subset at `results/bright_<subset>/caches/k2000.json`,
holding the top-2,000 hybrid-search document ids per query (no reranker
scores). Everything else (cross-encoder scores, run summaries, listwise
caches, analysis artifacts) is rebuilt by you, because it requires API keys
for the providers being measured.

Shipping the pools matters for reproducibility: collection **resumes into an
existing cache entry with `hybrid_order` held fixed**, so your collection
pass scores exactly the candidate sets the paper measured — no live Weaviate
retrieval, and no exposure to server-side index drift.

## API keys by stage

| Stage | Keys | Needed for |
|---|---|---|
| First stage (only if re-retrieving from scratch) | `WEAVIATE_URL`, `WEAVIATE_API_KEY` | `populate_db.py`, `hybrid_variance.py`, `latency_measurement.py` |
| Cross-encoder collection | `COHERE_API_KEY`, `VOYAGE_API_KEY`, `ZERANK_API_KEY` | `run_experiment.py --collect-only` |
| Listwise stage | `OPENAI_API_KEY` | `listwise_rerank.py`, `listwise_top100.py` |

Every analysis script is zero-network once its inputs exist under `results/`.

## The core loop

```bash
# 1. Fill the shipped k=2000 pool with cross-encoder scores (LIVE, resumable):
uv run python scripts/run_experiment.py --dataset biology --retrieved-k 2000 --collect-only

# 2. Derive the entire depth sweep offline (zero API calls):
uv run python scripts/k_sweep.py biology
```

Datasets: `biology`, `earth_science`, `economics`, `psychology`, `robotics`.
Collection is resumable per (provider, document) — interrupt and re-run
freely; only missing scores are fetched.

From there, [docs/reproducing.md](reproducing.md) maps every paper section to
its command, and [docs/data-formats.md](data-formats.md) documents every file
these steps read and write.

## Running the tests

```bash
uv run pytest tests/unit           # offline, no keys needed
uv run pytest -m integration       # live one-call-per-provider connectivity checks;
                                   # each provider skips if its key is absent
```
