# Reproducing the paper

Each command maps to the paper section it produces data for. Commands marked
**LIVE** spend API calls; everything else is a fast offline derivation over
`results/`. See [getting-started.md](getting-started.md) for keys and setup.

## The depth sweep (§ Scaling Retrieval, § Singleton Cross-Encoders)

```bash
# Fill the shipped k=2000 pool with scores, once per dataset (LIVE; resumable):
uv run python scripts/run_experiment.py --dataset biology --retrieved-k 2000 --collect-only

# Derive the full nested-prefix depth sweep (zero API calls):
uv run python scripts/k_sweep.py biology
```

Datasets: `biology`, `earth_science`, `economics`, `psychology`, `robotics`.
Derived runs land in `results/bright_<subset>/runs/k{N}_from_k2000.json`.

## Native-limit retrieval stability (Appendix: Retrieval Stability)

The nested-prefix sweep holds the initial ranking fixed; the robustness
check reruns native retrieval five times at each limit:

```bash
uv run python scripts/hybrid_variance.py --n-trials 5   # LIVE Weaviate calls
uv run python scripts/score_variance.py                 # reranker determinism
```

## Reranked depth and the capture ceiling (§ Reranked Depth)

Retaining 100 instead of 20 candidates from the cross-encoder, to measure
Recall@50/@100 against the first-stage ceiling:

```bash
uv run python scripts/k_sweep.py biology --reranked-k 100   # -> runs_rk100/
uv run python scripts/success_at_20.py
uv run python scripts/singleton_deep_recall.py
```

## The listwise stage (§ Singleton Listwise Rerankers, § Extended Window)

```bash
uv run python scripts/listwise_rerank.py --build-all-pools   # offline pool build
uv run python scripts/listwise_rerank.py --all-domains --model <model> --dry-run  # cost preflight, $0
uv run python scripts/listwise_rerank.py --all-domains --model <model>            # LIVE, 3 trials/query
uv run python scripts/listwise_top100.py -h                  # the n=100 window extension
```

Every (query, trial) ranking is cached and resumable — re-runs only pay for
missing calls. Models without a `MODEL_PRICES` entry require explicit
`--price-in/--price-out` ($/1M tokens).

## Width: fusion and disagreement (§ Exploring Width in Reranking)

```bash
uv run python scripts/equal_weight.py               # equal-weight fusion vs singletons (CE stage)
uv run python scripts/unique_successes.py           # per-model unique rank-1 successes (CE stage)
uv run python scripts/agreement.py --all-datasets   # rank-1 agreement / decorrelation
uv run python scripts/listwise_fusion.py            # equal-weight RRF at the listwise stage
uv run python scripts/listwise_unique_successes.py  # unique successes, listwise tier
```

## Routing headroom and its null (§ Discussion — query-dependent routing)

The oracle ceilings behind the routing-as-future-work claim, and the
winner's-curse controls that keep them honest:

```bash
uv run python scripts/oracle_config.py --k 2000        # routing vs blending decomposition
uv run python scripts/noise_null.py --singleton-only   # selection-on-noise null (CE)
uv run python scripts/listwise_oracle_routing.py       # selection ceiling, listwise tier
uv run python scripts/listwise_self_oracle.py          # self-ensemble control, listwise tier
```

## Deployment cost (§ The Cost of Depth, Width, and Stages)

```bash
uv run python scripts/latency_measurement.py --dataset biology   # LIVE timing calls
```

## Reading the numbers

- **Fusion menu is equal-weight only**
  (`scaling_reranked_retrieval.domain.conditions.CONDITIONS`): baselines, the
  three singletons, and equal-weight pair/3-way RRF+RSF blends. The paper's
  width result is about untuned, training-free fusion; no tilted weights
  exist in the code.
- **RSF ties.** RSF fusion produces exact score ties whose break order is
  `PYTHONHASHSEED`-dependent (~1 query of wobble); `scripts/noise_null.py`
  pins the seed for its bit-reproducible sweep. Differences below ~0.01 on a
  single RSF cell are noise.
- With ~100 queries per subset, 0.01 on a hit-based metric ≈ one query on a
  subset (≈ one query per subset for a cross-subset mean) — treat sub-0.02
  deltas accordingly.
