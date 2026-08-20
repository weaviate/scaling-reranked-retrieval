"""Listwise LLM reranking over a cross-encoder candidate pool (CE → listwise).

Pipeline per query (see mixture-of-rerankers/CLAUDE.md for the MoCE experiment):
  1. Hybrid search returns the top FIRST_STAGE_K=2000 candidates (from the cache).
  2. A pool-source condition ranks all 2000 and its top POOL_K become the
     candidate pool. Default: `zerank_only` top-20 (the Zerank-2 singleton
     order). `--pool-source voyage_only` reproduces the earlier Voyage-pool
     runs; `--pool-source rsf_equal_3way --pool-k 100` the original MoCE pool.
  3. A single-pass listwise LLM reranks that pool, N_TRIALS=3 independent times
     (LLM sampling is stochastic; trials give a mean ± std variance estimate).

Execution is two-phase: phase 1 runs every uncached (query, trial) LLM call
CONCURRENTLY (AsyncOpenAI + asyncio.Semaphore; --concurrency, default 8, 1 =
serial), appending each result to the per-query JSONL cache as it lands — so a
crash/interrupt resumes and only missing calls are re-paid. Phase 2 is the
unchanged serial aggregation loop reading everything from the (now-complete)
cache: ordered per-query progress, rescue ledger, and cost accounting need no
concurrency-awareness. Per-call latency_s stays per-call (it can read slightly
higher under rate-limit queueing); the fill phase's wall-clock is recorded
separately in listwise.json["latency"]["fill_wall_clock_s"].

The pool-source top-POOL_K is the fixed pool, and its order is the pointwise
baseline (`pool_baseline`) the listwise reranker must beat on the head metrics
(R@1/R@5; R@POOL_K is the pool ceiling, invariant under reordering). Question:
does listwise reasoning lift the head over the pointwise CE ranking?

ZERO new cross-encoder API calls: the three CE scores come from the k2000 cache;
the RSF fusion is recomputed locally. Only the listwise LLM is called.

Candidate-pool materialization (results/listwise/pools/):
  For each BRIGHT subset, the pool-source top-POOL_K-of-2000 pool (ids + texts +
  gold) is built ONCE from caches/k2000.json + corpus and saved to
    results/listwise/pools/<domain>__<pool-slug>__first<K>__top<P>.json
  (self-contained). Runs load that small file and skip BOTH the 50 MB cache and
  the corpus. Build all five with --build-all-pools; --rebuild-pool forces it.

  Determinism: singleton pools (e.g. voyage_only) sort distinct floats — exactly
  DerivedSearchAgent's singleton path, bit-reproducible. RSF pools: Derived-
  SearchAgent's RSF path tie-breaks via set() iteration (PYTHONHASHSEED-
  dependent; CLAUDE.md); here the fusion is reimplemented with identical math
  but iterating the hybrid-ordered pool, so ties break by hybrid rank → the
  saved pool is reproducible. A build-time guard checks the top-POOL_K set
  against DerivedSearchAgent (tolerating the documented ~1-doc tie wobble;
  singleton pools must match exactly).

Trials + outputs: each query is reranked --trials times (default 3); every
(query, trial) LLM response is cached to JSONL, so re-runs and trial-count
increases only pay for the missing trials. Per-domain outputs + the cross-
subset summary land under
  results/listwise/runs/<pool-slug>__first<K>__top<P>/<model>__<effort>/
so different pool configs, models, and reasoning efforts never clobber each
other — the run-dir identity matches the JSONL cache filename key exactly.
The cached per-(query, trial) rankings are the input to the RRF fusion
analysis in listwise_fusion.py (loaded via
src.application.listwise.load_listwise_rankings).

LLM: one model per invocation via the OpenAI SDK directly (default
`gpt-5.4-mini`; the experiment trio is EXPERIMENT_MODELS = gpt-5.4-mini +
gpt-5.6-luna + gpt-5.6-terra, all at effort "none" — run each, then analyze
with analysis/listwise_{fusion,unique_successes,oracle_routing}.py),
Chat Completions with strict structured outputs (`response_format` = json_schema
`{"ranking": [int]}`; position-encoded integers mapped back to doc_ids). Minimal
schema keeps output tokens small; reasoning tokens bill at the output price, so
reasoning effort is controlled and reasoning tokens are counted in the cost.

A pre-flight cost estimate prints BEFORE any LLM call. --dry-run does everything
except the LLM calls (so you can sanity-check the cost for $0).

ENV: OPENAI_API_KEY (real run only). No reranker keys are used.

CLI:
  uv run python scripts/listwise_rerank.py --build-all-pools   # one-time
  uv run python scripts/listwise_rerank.py --dry-run --n-queries 5
  uv run python scripts/listwise_rerank.py            # full biology (all queries)
  optional: --domain biology --model gpt-5.6-terra --first-stage-k 2000 --pool-k 20
            --pool-source zerank_only --trials 3 --reasoning-effort medium
            --price-in 2.50 --price-out 15.00 --reasoning-tokens-per-call N
            --concurrency 8 --rebuild-pool --quiet
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import time
from pathlib import Path

import tiktoken
from query_agent_benchmarking.internal.adapters.dataset import (
    in_memory_dataset_loader,
)

from src.application.derived import DerivedSearchAgent
from src.domain.conditions import CONDITIONS
from src.config import (
    CACHE_K,
    DATASETS,
    MODEL_OVERRIDES,
    PROVIDERS,
    RESULTS_DIR,
)
from src.domain.metrics import metric as _metric
from src.application.queryset import load_and_validate

from src.adapters import qab

# qab>=0.7 guard (replaces the old importlib.metadata version check) +
# load_search_dataset memoization.
qab.setup()

# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #

DOMAIN = "biology"
BRIGHT_SUBSETS = ["biology", "earth_science", "economics", "psychology", "robotics"]
SEED = 42  # only used when --n-queries subsets the intersection

# Two-stage knobs: hybrid retrieves FIRST_STAGE_K, the pool-source condition
# reranks them and its top POOL_K becomes the listwise pool.
DEFAULT_FIRST_STAGE_K = 2000
DEFAULT_POOL_K = 20
DEFAULT_POOL_SOURCE = "zerank_only"  # condition (singleton or RSF fusion) that builds the pool
DEFAULT_TRIALS = 3  # independent LLM trials per query (variance estimate)
DEFAULT_CONCURRENCY = 8  # parallel LLM calls in the phase-1 fill (1 = serial)
POOL_SOURCE = DEFAULT_POOL_SOURCE  # module global; overridden from --pool-source in main()
# Pool sources supported by the local deterministic reimplementation below
# (singletons + RSF fusions; RRF would need its own derive path).
POOL_SOURCE_CHOICES = ("voyage_only", "cohere_only", "zerank_only", "rsf_equal_3way")
POOL_LABEL = {
    "voyage_only": "Voyage rerank-2.5 singleton",
    "cohere_only": "Cohere rerank-v4.0-pro singleton",
    "zerank_only": "Zerank-2 singleton",
    "rsf_equal_3way": "MoCE cohere+voyage+zerank equal-weight RSF",
}

CUTOFFS = ("recall_at_1", "recall_at_5", "recall_at_20")
METRIC_LABEL = {"recall_at_1": "R@1", "recall_at_5": "R@5", "recall_at_20": "R@20"}

# The experiment's model trio (see the listwise_* analyses, which consume the
# cached rankings). Runs are one model per invocation; run each. All three at
# reasoning effort "none" so observed differences reflect the base model —
# effort is a cache-key segment, so runs at other efforts land in separate
# cache files and never mix.
EXPERIMENT_MODELS = ("gpt-5.4-mini", "gpt-5.6-luna", "gpt-5.6-terra")
DEFAULT_MODEL = EXPERIMENT_MODELS[0]  # raw OpenAI SDK model id (no "openai/" prefix)

# Known $/1M-token prices, used to resolve --price-in/--price-out when not
# passed explicitly. Models absent from this table REQUIRE explicit price
# flags — a silently-wrong cost estimate is worse than a hard stop.
MODEL_PRICES = {
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.6-luna": (1.00, 6.00),
    "gpt-5.6-terra": (2.50, 15.00),
}
# "none" = reasoning OFF (the experiment default): the model ranks in a single
# non-reasoning pass, so completion tokens are just the ranking array.
# Supported on the gpt-5.1+ family; pass --reasoning-effort to re-enable.
DEFAULT_REASONING_EFFORT = "none"
# Per-call hidden-reasoning-token ALLOWANCE for the pre-flight estimate, keyed by
# effort. medium≈4000 is the MEASURED average from a gpt-5.4 single-pass smoke
# (not yet re-measured on the gpt-5.6 pair — override --reasoning-tokens-per-call
# if their reasoning budgets differ materially; only the ESTIMATE uses this)
# over the 100-DOC MoCE pool (~3957/call; reasoning was ~94% of output tokens —
# the real cost driver, and it scales with pool size). The estimate scales this
# by pool_k/100 (floor 300), so a 20-doc pool assumes ~800/call at medium.
# Override with --reasoning-tokens-per-call.
REASONING_ALLOWANCE = {"none": 0, "minimal": 400, "low": 1500, "medium": 4000,
                       "high": 7000}
REASONING_ALLOWANCE_POOL_K = 100  # pool size the allowances were measured at
REASONING_ALLOWANCE_FLOOR = 300
MAX_COMPLETION_TOKENS = 16000  # generous cap; only ACTUAL tokens bill

# --------------------------------------------------------------------------- #
# Prompt templates + structured-output schema (logged to prompts/)             #
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = (
    "You are RankGPT, an expert relevance ranker. You are given a query and a "
    "numbered list of passages. Rank the passages from most to least relevant "
    "to the query and return the passage numbers in descending order of "
    "relevance, as the `ranking` array. Every passage number must appear "
    "exactly once. Do not include any explanation or per-passage reasoning — "
    "return only the ranking."
)

USER_TEMPLATE = (
    "Query: {query}\n\n"
    "Passages ({n} total):\n{passages}\n\n"
    "Rank the {n} passages above by their relevance to the query and return all "
    "{n} passage numbers in descending order of relevance; every integer from 1 "
    "to {n} must appear exactly once."
)

# Strict structured-output schema. Minimal on purpose: a single integer array,
# no per-doc objects/reasons (those would multiply output tokens). Root must be
# an object per OpenAI's structured-output rules, so the permutation lives under
# `ranking`. Strict mode guarantees valid JSON of this shape; permutation
# validity (uniqueness/completeness) is enforced by the recovery layer below.
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "listwise_ranking",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "ranking": {
                    "type": "array",
                    "description": "Passage numbers in descending relevance.",
                    "items": {"type": "integer"},
                }
            },
            "required": ["ranking"],
            "additionalProperties": False,
        },
    },
}


def build_user_prompt(query: str, doc_texts: list[str]) -> str:
    """Assemble the user message for a list of passages (input order = list order).
    Passage numbers are 1-based positions into the input list (RankGPT
    convention); the caller maps the returned permutation back to doc_ids."""
    passages = "\n".join(f"[{i}] {t}" for i, t in enumerate(doc_texts, 1))
    return USER_TEMPLATE.format(query=query, passages=passages, n=len(doc_texts))


# --------------------------------------------------------------------------- #
# Permutation parsing (structured-first, with recovery)                        #
# --------------------------------------------------------------------------- #

_ARRAY_RE = re.compile(r"\[.*\]", re.S)
_INT_RE = re.compile(r"-?\d+")


def extract_ints(text: str) -> list[int]:
    """Pull the ranked integer list: structured `{"ranking": [...]}` first, then
    any `[...]` array, then bare integers (recovery)."""
    s = (text or "").strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and isinstance(obj.get("ranking"), list):
            arr = obj["ranking"]
            if all(isinstance(v, (int, float)) for v in arr):
                return [int(v) for v in arr]
    except Exception:
        pass
    m = _ARRAY_RE.search(s)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list) and all(isinstance(v, (int, float)) for v in arr):
                return [int(v) for v in arr]
        except Exception:
            pass
    return [int(x) for x in _INT_RE.findall(s)]


def parse_permutation(text: str, n: int) -> tuple[list[int], bool]:
    """Parse a response into a 1-based ordering over [1, n]. Dedups, drops
    out-of-range, appends missing in input order (RankGPT recovery). parse_ok is
    False only when no valid identifier could be extracted."""
    nums = extract_ints(text)
    seen: set[int] = set()
    order: list[int] = []
    for x in nums:
        if 1 <= x <= n and x not in seen:
            seen.add(x)
            order.append(x)
    parse_ok = len(seen) > 0
    for i in range(1, n + 1):
        if i not in seen:
            order.append(i)
    return order, parse_ok


# --------------------------------------------------------------------------- #
# Tokenization (pre-flight estimate; input exact, output deterministic + est.) #
# --------------------------------------------------------------------------- #


def get_encoder(model: str):
    name = model.split("/")[-1]
    try:
        return tiktoken.encoding_for_model(name)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


def count_message_tokens(enc, system: str, user: str) -> int:
    return len(enc.encode(system)) + len(enc.encode(user)) + 2 * 4 + 3


def visible_perm_tokens(enc, n: int) -> int:
    """Token count of the compact JSON int list '[1, 2, ..., n]'."""
    return len(enc.encode(str(list(range(1, n + 1)))))


# --------------------------------------------------------------------------- #
# Fusion / ranking from cached scores (deterministic; zero API)                #
# --------------------------------------------------------------------------- #


def _min_max(scores: dict) -> dict:
    """Min-max normalize to [0,1]; matches derived.ScoreCache RSF (max→1.0 on a
    flat set)."""
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def _pool_scores(entry: dict, provider: str, pool: list[str]) -> dict:
    sc = entry.get(f"{provider}_scores", {}) or {}
    return {d: sc[d] for d in pool if d in sc}


def rsf_fused_ranking(entry: dict, rerankers, weights: dict, retrieved_k: int,
                      reranked_k: int) -> list[str]:
    """Equal-weight (or weighted) RSF fusion ranking of the top-`retrieved_k`
    hybrid pool, returning the top `reranked_k` doc_ids.

    Identical math to DerivedSearchAgent's RSF branch (per-reranker min-max →
    weighted sum), EXCEPT we build the fused dict iterating the hybrid-ordered
    pool (not set(pool)), so the stable sort breaks fused-score ties by hybrid
    rank → deterministic (no PYTHONHASHSEED dependence). This is the determinism
    fix CLAUDE.md endorses for the RSF path."""
    pool = entry["hybrid_order"][:retrieved_k]
    normed = {r: _min_max(_pool_scores(entry, r, pool)) for r in rerankers}
    fused: dict[str, float] = {}
    for d in pool:
        fused[d] = sum(weights.get(r, 0.0) * normed[r].get(d, 0.0) for r in rerankers)
    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return [d for d, _ in ranked[:reranked_k]]


def singleton_ranking(entry: dict, provider: str, retrieved_k: int,
                      reranked_k: int) -> list[str]:
    """Top-`reranked_k` of the top-`retrieved_k` hybrid pool by a single CE's
    score (deterministic: pool-ordered dict + stable sort)."""
    pool = entry["hybrid_order"][:retrieved_k]
    scores = _pool_scores(entry, provider, pool)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [d for d, _ in ranked[:reranked_k]]


def get_pool_condition():
    for c in CONDITIONS:
        if c.name == POOL_SOURCE:
            return c
    raise SystemExit(f"Condition {POOL_SOURCE!r} not found in run_experiment.CONDITIONS.")


def pool_source_ranking(entry: dict, cond, retrieved_k: int,
                        reranked_k: int) -> list[str]:
    """Rank the top-`retrieved_k` hybrid pool with the pool-source condition
    (singleton CE score sort, or weighted RSF fusion), deterministically."""
    if cond.provider in ("cohere", "voyage", "zerank"):
        return singleton_ranking(entry, cond.provider, retrieved_k, reranked_k)
    if cond.provider == "hybrid" and cond.fusion_method == "rsf":
        rerankers = cond.rerankers or PROVIDERS
        weights = cond.weights or {r: 1.0 / len(rerankers) for r in rerankers}
        return rsf_fused_ranking(entry, rerankers, weights, retrieved_k, reranked_k)
    raise SystemExit(
        f"Pool source {cond.name!r} unsupported: only singleton and RSF-fusion "
        "conditions have a local deterministic derive path here."
    )


# --------------------------------------------------------------------------- #
# Candidate-pool construction + materialization                                #
# --------------------------------------------------------------------------- #


def load_corpus_text_map(qab_name: str) -> dict[str, str]:
    corpus, _ = in_memory_dataset_loader(qab_name, corpus_only=True)
    return {d["dataset_id"]: d["content"] for d in corpus}


def pool_path(out_dir: Path, domain: str, first_k: int, pool_k: int) -> Path:
    slug = POOL_SOURCE.replace("_", "-")
    return out_dir / "pools" / f"{domain}__{slug}__first{first_k}__top{pool_k}.json"


def build_pool(domain: str, first_k: int, pool_k: int) -> dict:
    """Build the per-query pool-source top-`pool_k`-of-`first_k` pool over the
    all-three-present intersection (self-contained: ids + deduped texts + gold).
    Records the pool source's R@1 AND the three singletons' R@1 on the pool, plus
    a faithfulness guard vs DerivedSearchAgent. Zero CE API calls."""
    loaded = load_and_validate(domain)
    if loaded is None:
        raise SystemExit(f"No usable k{CACHE_K} cache for {domain}; cannot build pool.")
    cache, qs = loaded
    cond = get_pool_condition()
    print(f"  loading corpus texts for {DATASETS[domain].qab_name} ...")
    id2text_full = load_corpus_text_map(DATASETS[domain].qab_name)

    qlist = sorted(qs.gold.keys())
    n = len(qlist)
    queries: dict[str, dict] = {}
    doc_texts: dict[str, str] = {}
    r1 = {POOL_SOURCE: 0.0, **{p: 0.0 for p in PROVIDERS}}
    guard_exact = 0
    guard_maxdiff = 0
    for text in qlist:
        entry = cache.queries[text]
        gold = list(qs.gold[text])
        fused_ids = pool_source_ranking(entry, cond, first_k, pool_k)
        r1[POOL_SOURCE] += _metric("recall_at_1", gold, fused_ids)
        for p in PROVIDERS:
            r1[p] += _metric("recall_at_1", gold, singleton_ranking(entry, p, first_k, pool_k))
        # Faithfulness guard: compare top-pool_k SET vs DerivedSearchAgent.
        da = DerivedSearchAgent(cache=cache, retrieved_k=first_k, condition=cond,
                                reranked_k=pool_k)
        da_set = {o.object_id for o in da.run(text)}
        diff = len(set(fused_ids) ^ da_set) // 2
        guard_exact += int(diff == 0)
        guard_maxdiff = max(guard_maxdiff, diff)
        queries[text] = {
            "qid": qs.query_ids.get(text, text[:64]),
            "gold": sorted(qs.gold[text]),
            "doc_ids": fused_ids,
        }
        for d in fused_ids:
            if d not in doc_texts:
                doc_texts[d] = id2text_full[d]
    pool_r1 = {k: (r1[k] / n if n else 0.0) for k in r1}
    return {
        "metadata": {
            "domain": domain,
            "qab_dataset": DATASETS[domain].qab_name,
            "pool_source": POOL_SOURCE,
            "pool_provider": cond.provider,
            "rerankers": (list(cond.rerankers or PROVIDERS)
                          if cond.provider == "hybrid" else [cond.provider]),
            "weights": (dict(cond.weights) if cond.provider == "hybrid" and cond.weights
                        else None),
            "first_stage_k": first_k,
            "pool_k": pool_k,
            "cache_retrieved_k": CACHE_K,
            "model_overrides": MODEL_OVERRIDES,
            "intersection_n": n,
            "pool_r1": pool_r1,
            "fusion_guard": {
                "exact_topk_set_match_queries": guard_exact,
                "of_queries": n,
                "max_boundary_doc_diff": guard_maxdiff,
                "note": (f"vs DerivedSearchAgent {POOL_SOURCE}; small diffs are the "
                         "documented RSF set()-iteration tie wobble at the "
                         "rank-pool_k boundary, not a logic error. Singleton "
                         "pools must match exactly (distinct-float sort)."),
            },
            "description": (
                f"Per query: top {pool_k} docs of the {POOL_SOURCE} condition over "
                f"the top {first_k} hybrid candidates; doc_ids in ranked order. "
                "doc_texts = deduped union over all queries. Built from "
                "caches/k2000.json + corpus; zero CE API calls; deterministic."
            ),
        },
        "doc_texts": doc_texts,
        "queries": queries,
    }


def load_or_build_pool(out_dir: Path, domain: str, first_k: int, pool_k: int,
                       rebuild: bool) -> tuple[dict, Path]:
    path = pool_path(out_dir, domain, first_k, pool_k)
    if path.exists() and not rebuild:
        print(f"Loading cached candidate pool: {path}")
        with open(path) as f:
            return json.load(f), path
    print(f"Building {POOL_SOURCE} pool (top-{pool_k} of {first_k} hybrid) for "
          f"{domain} from cache + corpus (one-time) ...")
    payload = build_pool(domain, first_k, pool_k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)
    print(f"Saved pool to {path} ({len(payload['queries'])} queries, "
          f"{len(payload['doc_texts'])} unique docs).")
    return payload, path


def build_all_pools(out_dir: Path, first_k: int, pool_k: int, rebuild: bool) -> None:
    """Build + save the pool-source pools for all five BRIGHT subsets."""
    rows = []
    for ds in BRIGHT_SUBSETS:
        print(f"\n=== {ds} ===")
        payload, path = load_or_build_pool(out_dir, ds, first_k, pool_k, rebuild)
        m = payload["metadata"]
        r1 = m["pool_r1"]
        best_singleton = max((r1[p] for p in PROVIDERS))
        g = m["fusion_guard"]
        rows.append((ds, m["intersection_n"], r1[POOL_SOURCE], best_singleton,
                     g["exact_topk_set_match_queries"], g["of_queries"],
                     g["max_boundary_doc_diff"],
                     path.stat().st_size / 1e6))
    print(f"\n=== POOL BUILD SUMMARY ({POOL_SOURCE} top-"
          f"{pool_k} of {first_k}) ===")
    print(f"  {'subset':<14} {'n':>4} {'pool R@1':>9} {'best_sgl':>9} "
          f"{'lift':>7} {'guard':>10} {'maxdiff':>7} {'MB':>6}")
    for ds, n, rsf3, bs, ex, of, md, mb in rows:
        print(f"  {ds:<14} {n:>4} {rsf3:>9.3f} {bs:>9.3f} {rsf3-bs:>+7.3f} "
              f"{ex:>4}/{of:<5} {md:>7} {mb:>6.1f}")
    print(f"\n  Pools saved under {out_dir / 'pools'}/. 'guard' = queries whose "
          "top-k set exactly matches DerivedSearchAgent; maxdiff = worst boundary "
          f"tie wobble (docs). lift = {POOL_SOURCE} R@1 − best singleton R@1.")


# --------------------------------------------------------------------------- #
# Recall + interpretability helpers                                            #
# --------------------------------------------------------------------------- #


def first_gold_rank(ranked: list[str], gold: set) -> "int | None":
    for i, d in enumerate(ranked, 1):
        if d in gold:
            return i
    return None


def fmt_rank(r) -> str:
    return "miss" if r is None else f"r{r}"


def progress(msg: str = "") -> None:
    print(msg, flush=True)


def recall_row(ranked_by_query: dict[str, list[str]],
               gold_map: dict[str, set]) -> dict[str, float]:
    out = {c: 0.0 for c in CUTOFFS}
    n = len(ranked_by_query)
    for text, ranked in ranked_by_query.items():
        gold = list(gold_map[text])
        for c in CUTOFFS:
            out[c] += _metric(c, gold, ranked)
    return {c: (out[c] / n if n else 0.0) for c in CUTOFFS}


# --------------------------------------------------------------------------- #
# LLM call wrapper (OpenAI SDK directly; structured outputs)                    #
# --------------------------------------------------------------------------- #


class LLMConfig:
    def __init__(self, client, model: str, reasoning_effort: str,
                 max_completion_tokens: int):
        self.client = client  # openai.AsyncOpenAI (all LLM calls go through the async fill phase)
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_completion_tokens = max_completion_tokens


# --------------------------------------------------------------------------- #
# Per-query response cache (incremental + resumable; never re-pay a query)      #
# --------------------------------------------------------------------------- #
# One JSONL file per (domain, model, effort, pool config). Each line is one
# (query, trial) listwise result: {qid, trial, listwise_doc_ids, prompt/
# completion/reasoning tokens, latency_s, parse_failed}. Lines are appended as
# calls complete, so a crashed/partial run resumes, and a later run with more
# trials reuses the existing ones (entries missing "trial" are trial 0 — the
# pre-trials cache format). A cached entry is only honored if its doc set
# matches the current pool (guards against a rebuilt pool).


def query_cache_path(listwise_dir: Path, domain: str, model_id: str, effort: str,
                     first_k: int, pool_k: int) -> Path:
    mslug = model_id.replace("/", "-")
    pslug = POOL_SOURCE.replace("_", "-")
    return (listwise_dir / "cache" /
            f"{domain}__{mslug}__{effort}__{pslug}__first{first_k}__top{pool_k}.jsonl")


def load_query_cache(path: Path) -> dict:
    out: dict[tuple[str, int], dict] = {}
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if "qid" in e and "listwise_doc_ids" in e and "trial" in e:
                # last write wins (resume-safe)
                out[(str(e["qid"]), int(e["trial"]))] = e
    return out


def append_query_cache(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def cache_hit(qcache: dict, qid: str, trial: int,
              pool_ids: list[str]) -> "dict | None":
    """Return the cached entry for (qid, trial) iff its doc set matches the
    current pool."""
    e = qcache.get((str(qid), int(trial)))
    if e is not None and set(e["listwise_doc_ids"]) == set(pool_ids):
        return e
    return None


async def call_llm(cfg: LLMConfig, system: str, user: str) -> tuple[str, dict, float]:
    """One Chat Completions call with strict structured outputs; returns
    (text, normalized usage dict, latency_s). temperature is omitted (GPT-5
    reasoning models accept only the default)."""
    t0 = time.perf_counter()
    resp = await cfg.client.chat.completions.create(
        model=cfg.model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format=RESPONSE_FORMAT,
        reasoning_effort=cfg.reasoning_effort,
        max_completion_tokens=cfg.max_completion_tokens,
    )
    dt = time.perf_counter() - t0
    text = resp.choices[0].message.content or ""
    u = resp.usage
    details = getattr(u, "completion_tokens_details", None)
    usage = {
        "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
        "completion_tokens_details": {
            "reasoning_tokens": (getattr(details, "reasoning_tokens", 0) or 0)
            if details is not None else 0
        },
    }
    return text, usage, dt


def _usage_fields(u: dict) -> tuple[int, int, int]:
    details = u.get("completion_tokens_details") or {}
    if not isinstance(details, dict):
        details = getattr(details, "__dict__", {}) or {}
    return (int(u.get("prompt_tokens", 0) or 0),
            int(u.get("completion_tokens", 0) or 0),
            int(details.get("reasoning_tokens", 0) or 0))


async def rerank_single_pass(cfg: LLMConfig, query: str,
                             doc_ids: list[str],
                             id2text: dict[str, str]) -> dict:
    """Single-pass listwise rerank of the whole pool with one LLM call (+ one
    retry on a true parse failure; the retry stays sequential WITHIN this
    task). Returns a per-query record: reordered doc_ids, parse_failed, token
    counts (summed over calls), latency_s, and a status note."""
    texts = [id2text[d] for d in doc_ids]
    n = len(doc_ids)
    user = build_user_prompt(query, texts)
    text, u, dt = await call_llm(cfg, SYSTEM_PROMPT, user)
    pt, ct, rt = _usage_fields(u)
    order, ok = parse_permutation(text, n)
    note = "ok"
    if not ok:
        text, u2, dt2 = await call_llm(cfg, SYSTEM_PROMPT, user)
        pt2, ct2, rt2 = _usage_fields(u2)
        pt += pt2; ct += ct2; rt += rt2; dt += dt2
        order, ok = parse_permutation(text, n)
        note = "ok(retry)" if ok else "PARSE-FAIL→input order"
    parse_failed = not ok
    if parse_failed:
        order = list(range(1, n + 1))
    return {
        "doc_ids": [doc_ids[i - 1] for i in order],
        "parse_failed": parse_failed,
        "prompt_tokens": pt, "completion_tokens": ct, "reasoning_tokens": rt,
        "latency_s": round(dt, 3), "note": note,
    }


async def fill_query_cache(prep: dict, cfg: LLMConfig, args,
                           verbose: bool) -> set[tuple[str, int]]:
    """Phase 1: run every uncached (query, trial) LLM call concurrently
    (Semaphore-capped at --concurrency) and append each result to the JSONL
    cache as it completes. Returns the set of (qid, trial) keys filled this
    run (phase 2 counts exactly those as fresh spend).

    Failure containment: one call's terminal error (after the SDK's own
    retries) doesn't cancel the rest — completed entries are already on disk,
    so a re-run only pays for the failures. If any call failed we exit after
    the fill with a summary instead of aggregating incomplete results."""
    qmap, id2text = prep["qmap"], prep["id2text"]
    jobs = [(s, tr) for s in prep["sampled"] for tr in range(args.trials)
            if cache_hit(prep["qcache"], s["qid"], tr,
                         qmap[s["text"]]["doc_ids"]) is None]
    if not jobs:
        return set()
    concurrency = max(1, args.concurrency)
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()  # serializes JSONL appends + progress counters
    done = 0
    run_usd = 0.0
    t0 = time.perf_counter()
    progress(f"  fill: {len(jobs)} LLM calls at concurrency {concurrency} ...")

    async def one(s: dict, tr: int) -> tuple[str, int]:
        nonlocal done, run_usd
        async with sem:
            res = await rerank_single_pass(cfg, s["query"],
                                           qmap[s["text"]]["doc_ids"], id2text)
        entry = {"qid": s["qid"], "trial": tr,
                 "listwise_doc_ids": res["doc_ids"],
                 "prompt_tokens": res["prompt_tokens"],
                 "completion_tokens": res["completion_tokens"],
                 "reasoning_tokens": res["reasoning_tokens"],
                 "latency_s": res["latency_s"],
                 "parse_failed": res["parse_failed"]}
        async with lock:
            append_query_cache(prep["qcache_path"], entry)
            prep["qcache"][(str(s["qid"]), tr)] = entry
            done += 1
            run_usd += (res["prompt_tokens"] * args.price_in
                        + res["completion_tokens"] * args.price_out) / 1e6
            if verbose:
                progress(f"    [{done}/{len(jobs)}] q{s['qid']} trial {tr} · "
                         f"{res['latency_s']:5.1f}s · reason "
                         f"{res['reasoning_tokens']:>5,} · ${run_usd:.3f} so far"
                         + ("" if res["note"] == "ok" else f" · {res['note']}"))
        return (str(s["qid"]), tr)

    outcomes = await asyncio.gather(*(one(s, tr) for s, tr in jobs),
                                    return_exceptions=True)
    wall = time.perf_counter() - t0
    prep["fill_wall_s"] = round(wall, 1)
    failures = [(jobs[i], o) for i, o in enumerate(outcomes)
                if isinstance(o, BaseException)]
    progress(f"  fill done: {len(jobs) - len(failures)}/{len(jobs)} calls in "
             f"{wall:.0f}s wall (concurrency {concurrency}).")
    if failures:
        for (s, tr), exc in failures[:10]:
            progress(f"    FAILED q{s['qid']} trial {tr}: {exc!r}")
        raise SystemExit(
            f"{len(failures)} of {len(jobs)} LLM calls failed after SDK "
            f"retries; completed calls are cached in {prep['qcache_path']} — "
            "re-run to pay only for the missing ones."
        )
    return {o for o in outcomes if not isinstance(o, BaseException)}


# --------------------------------------------------------------------------- #
# Pre-flight cost estimate (single-pass only)                                   #
# --------------------------------------------------------------------------- #


def estimate_costs(enc, sampled, qmap, id2text, price_in, price_out,
                   reasoning_per_call) -> dict:
    """Tokenize each single-pass prompt; sum estimated cost. INPUT is exact (for
    the encoder) + json_schema overhead; OUTPUT = permutation length + per-call
    reasoning allowance (the cost driver, the main uncertainty)."""
    schema_tok = len(enc.encode(json.dumps(RESPONSE_FORMAT)))
    in_counts, out_counts = [], []
    for s in sampled:
        ids = qmap[s["text"]]["doc_ids"]
        user = build_user_prompt(s["query"], [id2text[d] for d in ids])
        in_counts.append(count_message_tokens(enc, SYSTEM_PROMPT, user) + schema_tok)
        out_counts.append(visible_perm_tokens(enc, len(ids)))
    calls = len(in_counts)
    in_tok = sum(in_counts)
    reason = reasoning_per_call * calls
    out_tok = sum(out_counts) + reason
    cost = (in_tok * price_in + out_tok * price_out) / 1e6
    return {
        "n_calls": calls,
        "input_tokens": in_tok,
        "est_visible_output_tokens": sum(out_counts),
        "est_reasoning_tokens": reason,
        "est_output_tokens": out_tok,
        "estimated_usd": cost,
        "assumptions": {
            "price_in_per_1m": price_in,
            "price_out_per_1m": price_out,
            "reasoning_tokens_per_call": reasoning_per_call,
            "note": ("Input exact (tiktoken) + json_schema overhead. Output = "
                     "permutation length + reasoning allowance/call (bills at "
                     "output price; the main uncertainty)."),
        },
    }


def print_estimate(est: dict) -> None:
    a = est["assumptions"]
    print("\n=== PRE-FLIGHT COST ESTIMATE (single-pass listwise) ===")
    print(f"  model prices: ${a['price_in_per_1m']:.2f}/1M in, "
          f"${a['price_out_per_1m']:.2f}/1M out; reasoning allowance "
          f"{a['reasoning_tokens_per_call']} tok/call")
    print(f"  calls {est['n_calls']} · input {est['input_tokens']:,} tok · "
          f"output {est['est_output_tokens']:,} tok "
          f"(visible {est['est_visible_output_tokens']:,} + reasoning "
          f"{est['est_reasoning_tokens']:,})")
    print(f"  estimated: ${est['estimated_usd']:.3f}")


# --------------------------------------------------------------------------- #
# Outputs                                                                       #
# --------------------------------------------------------------------------- #


def write_prompts(listwise_dir: Path) -> None:
    pdir = listwise_dir / "prompts"
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "system.txt").write_text(SYSTEM_PROMPT + "\n")
    (pdir / "user_template.txt").write_text(USER_TEMPLATE + "\n")
    (pdir / "response_format.json").write_text(json.dumps(RESPONSE_FORMAT, indent=2))


def write_plot(payload: dict, out_dir: Path) -> Path:
    """Grouped bar chart: pool baseline vs listwise trial-mean (± across-trial
    std error bars), R@1/5/20."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    res = payload["results"]
    labels = [f"{POOL_SOURCE} baseline (top-{payload['pool_k']})",
              f"Listwise single-pass (mean of {payload['trials']} trials)"]
    colors = ["#9e9e9e", "#1f77b4"]
    cutoffs = ["recall@1", "recall@5", "recall@20"]
    cut_labels = ["R@1", "R@5", "R@20"]

    x = np.arange(len(cutoffs))
    width = 0.4
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    base_vals = [res["pool_baseline"][c] for c in cutoffs]
    mean_vals = [res["listwise_mean"][c] for c in cutoffs]
    std_vals = [res["listwise_std"][c] for c in cutoffs]
    b1 = ax.bar(x - width / 2, base_vals, width, label=labels[0], color=colors[0])
    b2 = ax.bar(x + width / 2, mean_vals, width, yerr=std_vals, capsize=4,
                label=labels[1], color=colors[1])
    ax.bar_label(b1, fmt="%.3f", fontsize=8, padding=2)
    ax.bar_label(b2, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(cut_labels)
    ax.set_ylabel(f"Recall (mean over {payload['n_queries']} queries)")
    ax.set_ylim(0, max(0.05, max(base_vals + mean_vals)) * 1.25)
    ax.set_title(
        f"Listwise reranking of {POOL_SOURCE} top-{payload['pool_k']} "
        f"(of {payload['first_stage_k']} hybrid) — {payload['domain']}\n"
        f"{payload['model']} (effort={payload['reasoning_effort']}), "
        f"n={payload['n_queries']}, {payload['trials']} trials, seed "
        f"{payload['seed']}, actual ${payload['cost']['actual_usd']:.2f}"
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = out_dir / "listwise_plot.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def render_table(payload: dict) -> str:
    L: list[str] = []
    A = L.append
    res = payload["results"]
    lift = payload["lift_over_baseline"]["listwise_mean"]
    trials = payload["trials"]
    ceiling_key = f"recall@{payload['pool_k']}"
    base_keys = ["recall@1", "recall@5", "recall@20"]
    show_ceiling_col = ceiling_key not in base_keys
    pool_label = POOL_LABEL.get(POOL_SOURCE, POOL_SOURCE)
    A(f"# Listwise reranking of the {POOL_SOURCE} pool — {payload['domain']}")
    A("")
    A(f"Model `{payload['model']}` (effort {payload['reasoning_effort']}) · "
      f"domain {payload['domain']} · n={payload['n_queries']} (seed {payload['seed']}) "
      f"· {trials} independent trials per query (LLM sampling variance). "
      f"Pool = **{POOL_SOURCE} top-{payload['pool_k']} of "
      f"{payload['first_stage_k']} hybrid candidates** ({pool_label}); "
      "single-pass listwise reorders it.")
    A("")
    A(f"Cost: estimated ${payload['cost']['estimated_usd']:.3f} / actual "
      f"${payload['cost']['actual_usd']:.3f}. Pool file: `{payload['pool_file']}`.")
    A("")

    def _cell(v):
        return f"{v:.3f}" if isinstance(v, (int, float)) else "n/a"

    header = "| Condition | R@1 | R@5 | R@20 |"
    sep = "|---|---|---|---|"
    if show_ceiling_col:
        header += f" {ceiling_key.replace('recall@', 'R@')} (pool ceiling) |"
        sep += "---|"
    A(header)
    A(sep)

    def _row(label, r, std=None):
        cells = []
        for k in base_keys + ([ceiling_key] if show_ceiling_col else []):
            c = _cell(r.get(k))
            if std is not None and k in std:
                c += f" ± {std[k]:.3f}"
            cells.append(c)
        A(f"| {label} | " + " | ".join(cells) + " |")

    _row("`pool_baseline`", res["pool_baseline"])
    for tr, r in enumerate(res["listwise_trials"]):
        _row(f"`listwise` trial {tr}", r)
    _row(f"**`listwise` mean ± std ({trials} trials)**", res["listwise_mean"],
         std=res["listwise_std"])
    dcells = [f"{lift[k]:+.3f}" for k in base_keys]
    if show_ceiling_col:
        d = lift.get(ceiling_key)
        dcells.append(f"{d:+.3f}" if isinstance(d, (int, float)) else "—")
    A("| **Δ listwise mean − baseline** | " + " | ".join(dcells) + " |")
    A("")
    ceil_name = ceiling_key.replace("recall@", "R@")
    A(f"**{ceil_name} is the `{POOL_SOURCE}` top-{payload['pool_k']} pool "
      "ceiling** — the fraction of gold the pool source pulled into the candidate "
      "set the listwise reranker starts from. Listwise only reorders the fixed "
      f"{payload['pool_k']} docs, so {ceil_name} is invariant under it (identical "
      "across all rows, Δ=0) and upper-bounds the head metrics — gold beyond the "
      "pool is unreachable.")
    A("")
    pf = payload["parse_failures"]["total"]
    n_calls = payload["n_queries"] * trials
    A(f"Parse failures: {pf} of {n_calls} calls "
      f"(per trial: {payload['parse_failures']['per_trial']}).")
    A("")
    r1 = payload["pool_r1"]
    A("Pool R@1 (full intersection): "
      + ", ".join(f"{k} {r1[k]:.3f}" for k in [POOL_SOURCE] + list(PROVIDERS))
      + f". The `{POOL_SOURCE}` pointwise order is the baseline listwise must beat.")
    A("")
    # Go/no-go read (on the trial mean, with the across-trial std for scale).
    r1_lift = lift["recall@1"]
    r1_std = res["listwise_std"]["recall@1"]
    pf_rate = pf / n_calls if n_calls else 0.0
    go = r1_lift >= 0.04 and pf_rate < 0.10
    A("## Go / no-go read")
    A("")
    san = payload["baseline_sanity"]
    A(f"Regression sanity: `pool_baseline` R@1 over these {payload['n_queries']} "
      f"= {san['sampled']['recall@1']:.3f} vs full-{san['full_intersection_n']} "
      f"= {san['full_intersection']['recall@1']:.3f}.")
    A("")
    verdict = "**GO**" if go else "**NO-GO**"
    A(f"{verdict}. Single-pass listwise mean R@1 lift over the `{POOL_SOURCE}` "
      f"pointwise order is {r1_lift:+.3f} (~{r1_lift*payload['n_queries']:.0f} "
      f"queries on {payload['n_queries']}; across-trial std {r1_std:.3f}); "
      f"parse-failure rate {pf_rate:.1%}. "
      + ("Mean lift clears ~+0.04 (≳2 queries) with low parse failures — "
         "listwise on top of the CE pool pays off; spec the full sweep."
         if go else
         f"Listwise does not clear ~+0.04 mean R@1 over the pointwise "
         f"`{POOL_SOURCE}` order (or parse failures are high) — the listwise "
         "layer doesn't pay for its cost/latency here.")
      )
    A("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Driver                                                                        #
# --------------------------------------------------------------------------- #


def _as_out(d: dict) -> dict:
    return {"recall@1": d["recall_at_1"], "recall@5": d["recall_at_5"],
            "recall@20": d["recall_at_20"]}


def prepare_domain(listwise_dir: Path, domain: str, args, enc,
                   reasoning_per_call: int, model_id: str) -> dict:
    """Load (or build) the pool, sample, load the per-query cache, and pre-flight
    estimate ONLY the uncached queries (so the estimate reflects what will pay)."""
    pools, ppath = load_or_build_pool(listwise_dir, domain, args.first_stage_k,
                                      args.pool_k, args.rebuild_pool)
    meta = pools["metadata"]
    qmap = pools["queries"]
    id2text = pools["doc_texts"]
    gold_map = {t: set(v["gold"]) for t, v in qmap.items()}
    intersection = sorted(qmap.keys())
    if args.n_queries is None:  # default: the full intersection, in sorted order
        texts = list(intersection)
    else:                       # subset: deterministic seed-42 sample
        n_req = min(args.n_queries, len(intersection))
        if n_req < args.n_queries:
            print(f"[warn] {domain}: requested {args.n_queries} but intersection "
                  f"has {len(intersection)}; sampling {n_req}.")
        texts = random.Random(SEED).sample(intersection, n_req)
    n = len(texts)
    sampled = [{"text": t, "query": t, "qid": qmap[t]["qid"]} for t in texts]

    qcache_path = query_cache_path(listwise_dir, domain, model_id,
                                   args.reasoning_effort, args.first_stage_k,
                                   args.pool_k)
    qcache = load_query_cache(qcache_path) if args.skip_existing else {}
    # One LLM call per (query, trial); estimate only the uncached pairs.
    to_compute = [s for s in sampled for tr in range(args.trials)
                  if cache_hit(qcache, s["qid"], tr,
                               qmap[s["text"]]["doc_ids"]) is None]
    n_calls_total = n * args.trials
    n_cached = n_calls_total - len(to_compute)
    est = estimate_costs(enc, to_compute, qmap, id2text, args.price_in,
                         args.price_out, reasoning_per_call)
    return {"domain": domain, "ppath": ppath, "meta": meta, "qmap": qmap,
            "id2text": id2text, "gold_map": gold_map, "intersection": intersection,
            "texts": texts, "sampled": sampled, "n": n, "est": est,
            "qcache": qcache, "qcache_path": qcache_path, "n_cached": n_cached,
            "n_calls_total": n_calls_total, "model_id": model_id}


RESCUE_QUERY_TRUNC = 120


def build_rescue_ledger(qid: str, text: str, gold: set, pool_ids: list[str],
                        trial_listwise_ids: list[list[str]],
                        head_cut: int) -> dict:
    """Per-query rescue ledger: for each gold doc in the top-pool, its baseline
    (pool-source) rank and its listwise rank in EVERY trial, plus per-query
    roll-ups (mean over trials).

    Each trial's ids are a permutation of pool_ids (single-pass reorders the
    whole pool; recovery appends any dropped ids), so every gold-in-pool doc has
    a rank in every trial. `rank` = the position of that specific gold doc
    (1-based). `head_cut` = the golds@H roll-up cutoff (5 for small pools where
    golds@pool_k would be invariant, else 20)."""
    base_rank = {d: i for i, d in enumerate(pool_ids, 1)}
    trial_ranks = [{d: i for i, d in enumerate(ids, 1)} for ids in trial_listwise_ids]
    golds_in_pool = [d for d in pool_ids if d in gold]
    per_gold = []
    for d in golds_in_pool:
        br = base_rank[d]
        lrs = [tr.get(d) for tr in trial_ranks]
        deltas = [br - lr for lr in lrs if lr is not None]
        mean_lr = (sum(lr for lr in lrs if lr is not None) /
                   len([lr for lr in lrs if lr is not None])) if deltas else None
        per_gold.append({
            "doc_id": d, "baseline_rank": br, "listwise_ranks": lrs,
            "mean_listwise_rank": round(mean_lr, 2) if mean_lr is not None else None,
            "mean_delta": (round(sum(deltas) / len(deltas), 2) if deltas else None),
        })
    per_gold.sort(key=lambda g: (g["mean_listwise_rank"] is None,
                                 g["mean_listwise_rank"] or 0))
    base_ranks = [g["baseline_rank"] for g in per_gold]
    first_by_trial = []
    heads_by_trial = []
    for tr in trial_ranks:
        rs = [tr[d] for d in golds_in_pool if d in tr]
        first_by_trial.append(min(rs) if rs else None)
        heads_by_trial.append(sum(1 for r in rs if r <= head_cut))
    mean_deltas = [g["mean_delta"] for g in per_gold if g["mean_delta"] is not None]
    return {
        "qid": qid,
        "query": text[:RESCUE_QUERY_TRUNC],
        "n_gold": len(gold),
        "golds_in_pool": len(golds_in_pool),
        "baseline_first_gold_rank": min(base_ranks) if base_ranks else None,
        "listwise_first_gold_rank_by_trial": first_by_trial,
        "head_cut": head_cut,
        "baseline_golds_in_head": sum(1 for r in base_ranks if r <= head_cut),
        "listwise_golds_in_head_by_trial": heads_by_trial,
        "largest_rescue": max(mean_deltas) if mean_deltas else 0,
        "per_gold": per_gold,
    }


def render_rescues_table(payload: dict) -> str:
    """rescues_table.md, sorted by largest mean single-gold rescue."""
    L: list[str] = []
    A = L.append
    trials = payload["trials"]
    head_cut = payload["rescues"][0]["head_cut"] if payload["rescues"] else 5
    A(f"# Listwise rescue ledger — {payload['domain']}")
    A("")
    A("Per-query gold movement; **rank = the position of that specific gold doc**"
      f" (1-based). Base = `{POOL_SOURCE}` top-{payload['pool_k']} pool order; LW "
      f"= after single-pass listwise, one column entry per trial ({trials} "
      "trials, '/' separated). Δ̄ = mean over trials of (base rank − LW rank); "
      "positive = rescued upward. Sorted by the largest mean single-gold rescue. "
      f"n={payload['n_queries']}, {payload['model']} (effort "
      f"{payload['reasoning_effort']}).")
    A("")
    A(f"| qid | query | gold n/in-pool | first-gold base→LW | golds@{head_cut} "
      "base→LW | best Δ̄ | lat s | per-gold base→LW trials (Δ̄), by mean LW rank |")
    A("|---|---|---|---|---|---|---|---|")
    rows = sorted(payload["rescues"], key=lambda r: r["largest_rescue"], reverse=True)
    for r in rows:
        q = r["query"][:60].replace("|", "/").replace("\n", " ").strip()
        pg = "; ".join(
            f"{g['baseline_rank']}→" +
            "/".join(fmt_rank(lr) if lr is None else str(lr)
                     for lr in g["listwise_ranks"]) +
            (f" ({g['mean_delta']:+.1f})" if g["mean_delta"] is not None else " (miss)")
            for g in r["per_gold"]
        )
        first_lw = "/".join(fmt_rank(x) for x in r["listwise_first_gold_rank_by_trial"])
        heads_lw = "/".join(str(x) for x in r["listwise_golds_in_head_by_trial"])
        lat = r.get("latency_s")
        lat_s = f"{lat:.1f}" if isinstance(lat, (int, float)) else "n/a"
        A(f"| {r['qid']} | {q} | {r['n_gold']}/{r['golds_in_pool']} "
          f"| {fmt_rank(r['baseline_first_gold_rank'])}→{first_lw} "
          f"| {r['baseline_golds_in_head']}→{heads_lw} "
          f"| {r['largest_rescue']:+.1f} | {lat_s} | {pg} |")
    A("")
    return "\n".join(L)


def run_domain_listwise(prep: dict, args, run_dir: Path, verbose: bool,
                        fresh_keys: set[tuple[str, int]]) -> dict:
    """Phase 2: aggregate one prepared domain from the (now-complete) per-query
    cache — zero LLM calls here; every (query, trial) was filled by
    fill_query_cache or a prior run. `fresh_keys` marks the entries paid for
    this run (cost accounting). Writes per-domain outputs; returns a compact
    summary for the cross-subset table."""
    domain = prep["domain"]
    qmap, gold_map = prep["qmap"], prep["gold_map"]
    sampled, texts, intersection = prep["sampled"], prep["texts"], prep["intersection"]
    meta, est, n = prep["meta"], prep["est"], prep["n"]
    qcache, qcache_path = prep["qcache"], prep["qcache_path"]
    r1pool = meta["pool_r1"]
    trials = args.trials
    head_cut = 5 if args.pool_k <= 20 else 20  # golds@H roll-up for the ledger

    ranked_trials: list[dict[str, list[str]]] = [{} for _ in range(trials)]
    parse_fails_by_trial = [0] * trials
    rescues: list[dict] = []
    # Fresh = computed this run (what you pay now); cum = all calls incl. cached.
    fresh_in = fresh_out = fresh_reason = fresh_calls = 0
    cum_in = cum_out = cum_reason = 0
    n_cached = 0
    latencies: list[float] = []

    progress(f"\n=== {domain} === {n} queries × {trials} trials over the "
             f"{POOL_SOURCE} top-{args.pool_k} pool "
             f"({len(fresh_keys)} filled this run, "
             f"{prep['n_calls_total'] - len(fresh_keys)} previously cached).")
    for idx, s in enumerate(sampled, 1):
        t = s["text"]
        pool_ids = qmap[t]["doc_ids"]
        gold = gold_map[t]
        in_pool = sum(1 for d in pool_ids if d in gold)
        base_rank = first_gold_rank(pool_ids, gold)
        progress(f"\n[{idx}/{n}] q{s['qid']}  \"{t[:60].strip()}\"  "
                 f"(gold {len(gold)}, {in_pool} in pool, baseline first-gold "
                 f"{fmt_rank(base_rank)})")
        trial_ids: list[list[str]] = []
        q_lats: list[float] = []
        for tr in range(trials):
            tag = f"trial {tr}"
            key = (str(s["qid"]), tr)
            entry = qcache.get(key)
            if entry is None:  # fill_query_cache guarantees completeness
                raise SystemExit(
                    f"BUG: no cached result for q{s['qid']} trial {tr} after "
                    "the fill phase — aggregation cannot proceed."
                )
            if key in fresh_keys:
                fresh_in += int(entry.get("prompt_tokens", 0))
                fresh_out += int(entry.get("completion_tokens", 0))
                fresh_reason += int(entry.get("reasoning_tokens", 0))
                fresh_calls += 1
            else:
                n_cached += 1
            if verbose:
                src = "fresh " if key in fresh_keys else "cached"
                progress(f"    {tag:<7}: {src} · "
                         f"{entry.get('latency_s', 0):.1f}s · reason "
                         f"{int(entry.get('reasoning_tokens', 0)):,}")

            ids = entry["listwise_doc_ids"]
            ranked_trials[tr][t] = ids
            trial_ids.append(ids)
            parse_fails_by_trial[tr] += int(bool(entry.get("parse_failed", False)))
            cum_in += int(entry.get("prompt_tokens", 0))
            cum_out += int(entry.get("completion_tokens", 0))
            cum_reason += int(entry.get("reasoning_tokens", 0))
            lat = entry.get("latency_s")
            if lat is not None:
                latencies.append(float(lat))
                q_lats.append(float(lat))

        led = build_rescue_ledger(s["qid"], t, gold, pool_ids, trial_ids, head_cut)
        led["latency_s"] = (round(sum(q_lats) / len(q_lats), 3) if q_lats else None)
        rescues.append(led)
        run_cost = (fresh_in * args.price_in + fresh_out * args.price_out) / 1e6
        first_lw = "/".join(fmt_rank(x)
                            for x in led["listwise_first_gold_rank_by_trial"])
        heads_lw = "/".join(str(x) for x in led["listwise_golds_in_head_by_trial"])
        progress(f"    → first-gold base {fmt_rank(led['baseline_first_gold_rank'])}"
                 f" → listwise {first_lw} · golds@{head_cut} "
                 f"{led['baseline_golds_in_head']}→{heads_lw} · best Δ̄ "
                 f"{led['largest_rescue']:+.1f} | running ${run_cost:.3f} "
                 f"({fresh_calls} new, {n_cached} cached)")

    base_ranked = {t: qmap[t]["doc_ids"] for t in texts}
    ceiling_key = f"recall@{args.pool_k}"

    def _rceil(rbq):
        # Pool ceiling R@pool_k. Invariant under listwise (it only reorders the
        # pool), so baseline and every trial agree by construction; computed on
        # each so a mismatch would flag a dropped-doc bug.
        return (sum(_metric(f"recall_at_{args.pool_k}", list(gold_map[t]), rbq[t])
                    for t in rbq) / len(rbq)) if rbq else 0.0

    def _with_ceiling(rbq):
        out = _as_out(recall_row(rbq, gold_map))
        out[ceiling_key] = _rceil(rbq)  # == recall@20 when pool_k=20 (invariant)
        return out

    per_trial = [_with_ceiling(ranked_trials[tr]) for tr in range(trials)]
    keys = list(per_trial[0].keys())

    def _mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    def _std(vals):
        if len(vals) < 2:
            return 0.0
        m = _mean(vals)
        return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5

    results = {
        "pool_baseline": _with_ceiling(base_ranked),
        "listwise_trials": per_trial,
        "listwise_mean": {k: _mean([r[k] for r in per_trial]) for k in keys},
        "listwise_std": {k: _std([r[k] for r in per_trial]) for k in keys},
    }
    lift = {"listwise_mean": {k: results["listwise_mean"][k] -
                              results["pool_baseline"][k] for k in keys}}
    full_base = {t: qmap[t]["doc_ids"] for t in intersection}
    res_base_full = _as_out(recall_row(full_base, gold_map))
    parse_fails = sum(parse_fails_by_trial)

    actual_usd = (fresh_in * args.price_in + fresh_out * args.price_out) / 1e6
    cumulative_usd = (cum_in * args.price_in + cum_out * args.price_out) / 1e6
    lat_sorted = sorted(latencies)
    latency_block = {
        "n": len(latencies),
        "mean_s": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "median_s": round(lat_sorted[len(lat_sorted) // 2], 3) if latencies else None,
        "min_s": round(min(latencies), 3) if latencies else None,
        "max_s": round(max(latencies), 3) if latencies else None,
        "total_s": round(sum(latencies), 3) if latencies else None,
        "fill_wall_clock_s": prep.get("fill_wall_s"),  # this run's concurrent fill (None if fully cached)
        "concurrency": args.concurrency,
        "note": ("per-call single-pass listwise inference latency (s), pooled "
                 "over all trials; cached calls carry their originally-measured "
                 "latency. Calls run concurrently (fill phase), so wall-clock "
                 "= fill_wall_clock_s, not total_s; per-call latency can read "
                 "slightly high under rate-limit queueing."),
    }

    payload = {
        "model": prep["model_id"],
        "api": "openai_chat_completions",
        "response_format": "json_schema(strict): {ranking: [int]}",
        "domain": domain,
        "n_queries": n,
        "trials": trials,
        "seed": SEED,
        "pool_source": POOL_SOURCE,
        "first_stage_k": args.first_stage_k,
        "pool_k": args.pool_k,
        "pool_file": str(prep["ppath"]),
        "query_cache_file": str(qcache_path),
        "reasoning_effort": args.reasoning_effort,
        "sampled_query_ids": [s["qid"] for s in sampled],
        "pool_r1": r1pool,
        "results": results,
        "lift_over_baseline": lift,
        "baseline_sanity": {
            "full_intersection_n": len(intersection),
            "sampled": results["pool_baseline"],
            "full_intersection": res_base_full,
        },
        "latency": latency_block,
        "cost": {
            "estimated_usd": est["estimated_usd"],
            "actual_usd": actual_usd,            # this run (fresh calls only)
            "cumulative_usd": cumulative_usd,    # all queries incl. cached
            "input_tokens": fresh_in,
            "output_tokens": fresh_out,
            "reasoning_tokens": fresh_reason,
            "cumulative_input_tokens": cum_in,
            "cumulative_output_tokens": cum_out,
            "cumulative_reasoning_tokens": cum_reason,
            "n_new_calls": fresh_calls,
            "n_cached": n_cached,
            "price_in_per_1m": args.price_in,
            "price_out_per_1m": args.price_out,
            "estimate_detail": est,
        },
        "parse_failures": {"per_trial": parse_fails_by_trial, "total": parse_fails},
        "n_calls": fresh_calls,
        "rescues": rescues,
    }

    dom_dir = run_dir / domain
    dom_dir.mkdir(parents=True, exist_ok=True)
    (dom_dir / "listwise.json").write_text(json.dumps(payload, indent=2))
    (dom_dir / "listwise_table.md").write_text(render_table(payload))
    (dom_dir / "rescues_table.md").write_text(render_rescues_table(payload))
    write_plot(payload, dom_dir)

    rpc = fresh_reason / fresh_calls if fresh_calls else 0.0
    lat_str = f"{latency_block['median_s']:.1f}s/call median" if latencies else "n/a"
    print(f"  [{domain}] ${actual_usd:.3f} this run ({fresh_calls} new, {n_cached} "
          f"cached; cumulative ${cumulative_usd:.3f}) · reasoning ~{rpc:.0f}/call · "
          f"latency {lat_str} · pool R@1 {results['pool_baseline']['recall@1']:.3f} "
          f"→ listwise {results['listwise_mean']['recall@1']:.3f} "
          f"± {results['listwise_std']['recall@1']:.3f} "
          f"({lift['listwise_mean']['recall@1']:+.3f}) · parse-fail "
          f"{parse_fails}/{n * trials}")
    return {"domain": domain, "n": n, "trials": trials,
            "results": results, "lift": lift["listwise_mean"], "pool_r1": r1pool,
            "actual_usd": actual_usd}


def write_cross_subset_summary(summaries: list[dict], run_dir: Path,
                               args) -> None:
    """Aggregate per-subset pool-vs-listwise into a cross-subset table + plot."""
    cuts = ["recall@1", "recall@5", "recall@20"]
    clab = {"recall@1": "R@1", "recall@5": "R@5", "recall@20": "R@20"}

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    def agg(subs, key):  # key in {"pool","listwise","std","lift"}
        out = {}
        for c in cuts:
            if key == "pool":
                out[c] = mean([s["results"]["pool_baseline"][c] for s in subs])
            elif key == "listwise":
                out[c] = mean([s["results"]["listwise_mean"][c] for s in subs])
            elif key == "std":
                out[c] = mean([s["results"]["listwise_std"][c] for s in subs])
            else:
                out[c] = mean([s["lift"][c] for s in subs])
        return out

    payload = {
        "pool_source": POOL_SOURCE, "pool_k": args.pool_k,
        "first_stage_k": args.first_stage_k, "trials": args.trials,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "per_subset": {s["domain"]: {"n": s["n"],
                                     "results": s["results"], "lift": s["lift"],
                                     "actual_usd": s["actual_usd"]} for s in summaries},
        "aggregate_mean_all": {"pool": agg(summaries, "pool"),
                               "listwise": agg(summaries, "listwise"),
                               "listwise_std": agg(summaries, "std"),
                               "lift": agg(summaries, "lift")},
        "total_actual_usd": sum(s["actual_usd"] for s in summaries),
    }
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2))

    # Markdown table. LW cells are trial-mean ± across-trial std.
    L = [f"# Listwise over the {POOL_SOURCE} pool — cross-subset summary", "",
         f"`{args.model}` (effort {args.reasoning_effort}), single-pass listwise "
         f"over the {POOL_SOURCE} top-{args.pool_k}-of-{args.first_stage_k} pool, "
         f"{args.trials} trials per query (LW = mean ± across-trial std). "
         f"Total spend ${payload['total_actual_usd']:.2f}.", "",
         "| Subset | n | " + " | ".join(f"Pool {clab[c]}" for c in cuts) + " | "
         + " | ".join(f"LW {clab[c]}" for c in cuts) + " | "
         + " | ".join(f"Δ {clab[c]}" for c in cuts) + " |",
         "|---|---|" + "---|" * (3 * len(cuts))]
    for s in summaries:
        row = [s["domain"], str(s["n"])]
        row += [f"{s['results']['pool_baseline'][c]:.3f}" for c in cuts]
        row += [f"{s['results']['listwise_mean'][c]:.3f} ± "
                f"{s['results']['listwise_std'][c]:.3f}" for c in cuts]
        row += [f"{s['lift'][c]:+.3f}" for c in cuts]
        L.append("| " + " | ".join(row) + " |")
    a = payload["aggregate_mean_all"]
    row = ["**mean (all)**", ""]
    row += [f"{a['pool'][c]:.3f}" for c in cuts]
    row += [f"{a['listwise'][c]:.3f} ± {a['listwise_std'][c]:.3f}" for c in cuts]
    row += [f"{a['lift'][c]:+.3f}" for c in cuts]
    L.append("| " + " | ".join(row) + " |")
    L += ["", f"Δ = listwise trial-mean − {POOL_SOURCE} pointwise. Aggregates are "
          "mean across subsets (std cells: mean of per-subset stds).", ""]
    (run_dir / "summary_table.md").write_text("\n".join(L))

    # Plot: per-subset pool vs listwise R@1 (mean ± std).
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    doms = [s["domain"] for s in summaries]
    x = np.arange(len(doms))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(doms)), 5.2))
    b1 = ax.bar(x - w / 2, [s["results"]["pool_baseline"]["recall@1"] for s in summaries],
                w, label=f"{POOL_SOURCE} baseline R@1", color="#9e9e9e")
    b2 = ax.bar(x + w / 2, [s["results"]["listwise_mean"]["recall@1"] for s in summaries],
                w, yerr=[s["results"]["listwise_std"]["recall@1"] for s in summaries],
                capsize=4, label=f"Listwise R@1 (mean of {args.trials} trials)",
                color="#1f77b4")
    ax.bar_label(b1, fmt="%.2f", fontsize=8)
    ax.bar_label(b2, fmt="%.2f", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(doms)
    ax.set_ylabel("Recall@1 (mean over queries)")
    ax.set_title(f"Listwise vs {POOL_SOURCE} pointwise (R@1) — top-{args.pool_k} pool\n"
                 f"{args.model} (effort {args.reasoning_effort}, {args.trials} trials)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "summary_plot.png", dpi=150)
    plt.close(fig)

    print("\n=== CROSS-SUBSET SUMMARY (R@1) ===")
    for s in summaries:
        print(f"  {s['domain']:<14} pool {s['results']['pool_baseline']['recall@1']:.3f} "
              f"→ listwise {s['results']['listwise_mean']['recall@1']:.3f} "
              f"± {s['results']['listwise_std']['recall@1']:.3f} "
              f"({s['lift']['recall@1']:+.3f})")
    a = payload["aggregate_mean_all"]
    print(f"  mean (all): pool {a['pool']['recall@1']:.3f} → listwise "
          f"{a['listwise']['recall@1']:.3f} ({a['lift']['recall@1']:+.3f})")
    print(f"\nWrote {run_dir}/summary.json + summary_table.md + summary_plot.png")


def summary_from_payload(pl: dict) -> dict:
    """Reconstruct the cross-subset summary dict from a saved listwise.json."""
    return {"domain": pl["domain"], "n": pl["n_queries"],
            "trials": pl.get("trials", 1),
            "results": pl["results"],
            "lift": pl["lift_over_baseline"]["listwise_mean"],
            "pool_r1": pl["pool_r1"], "actual_usd": pl["cost"]["actual_usd"]}


def matching_existing(run_dir: Path, domain: str, prep: dict, args,
                      model_id: str) -> "dict | None":
    """Return a saved listwise.json payload if --skip-existing and it matches the
    current config (so the run can be reused instead of re-paid), else None."""
    if not args.skip_existing:
        return None
    p = run_dir / domain / "listwise.json"
    if not p.exists():
        return None
    try:
        pl = json.loads(p.read_text())
    except Exception:
        return None
    matches = (
        pl.get("model") == model_id
        and pl.get("n_queries") == prep["n"]
        and pl.get("trials") == args.trials
        and pl.get("seed") == SEED
        and pl.get("pool_source") == POOL_SOURCE
        and pl.get("first_stage_k") == args.first_stage_k
        and pl.get("pool_k") == args.pool_k
        and pl.get("reasoning_effort") == args.reasoning_effort
    )
    return pl if matches else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="Assemble prompts, tokenize, print the estimate. No LLM "
                         "calls; no key needed if the pool exists ($0).")
    ap.add_argument("--build-all-pools", action="store_true",
                    help="Build + save the selected --pool-source pools for all "
                         "5 BRIGHT subsets, then exit (one-time materialization).")
    ap.add_argument("--n-queries", type=int, default=None,
                    help="Subset size: seed-42 sample of this many intersection "
                         "queries. Default: the FULL intersection (all queries).")
    ap.add_argument("--domain", "--dataset", dest="domain", default=DOMAIN,
                    choices=BRIGHT_SUBSETS,
                    help="BRIGHT subset to run (alias: --dataset). Default biology.")
    ap.add_argument("--all-domains", action="store_true",
                    help="Run all 5 BRIGHT subsets and write a cross-subset summary.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--first-stage-k", type=int, default=DEFAULT_FIRST_STAGE_K)
    ap.add_argument("--pool-k", type=int, default=DEFAULT_POOL_K)
    ap.add_argument("--pool-source", default=DEFAULT_POOL_SOURCE,
                    choices=POOL_SOURCE_CHOICES,
                    help="Condition whose top-pool_k ranking is the listwise "
                         "candidate pool AND the pointwise baseline. Default "
                         f"{DEFAULT_POOL_SOURCE} (Zerank-2 singleton); "
                         "voyage_only reproduces the earlier Voyage-pool runs; "
                         "rsf_equal_3way the original MoCE pool.")
    ap.add_argument("--trials", type=int, default=DEFAULT_TRIALS,
                    help="Independent LLM trials per query (mean ± std variance "
                         f"estimate; LLM sampling is stochastic). Default "
                         f"{DEFAULT_TRIALS}.")
    ap.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT,
                    choices=("none", "minimal", "low", "medium", "high"),
                    help="GPT-5-family reasoning effort. Default 'none' = "
                         "reasoning OFF (the experiment setting); caches and "
                         "run dirs are keyed by this, so efforts never mix.")
    ap.add_argument("--price-in", type=float, default=None,
                    help="$/1M input tokens. Default: MODEL_PRICES entry for "
                         "--model; required if the model has no entry.")
    ap.add_argument("--price-out", type=float, default=None,
                    help="$/1M output tokens. Default: MODEL_PRICES entry for "
                         "--model; required if the model has no entry.")
    ap.add_argument("--reasoning-tokens-per-call", type=int, default=None,
                    help="Override the per-call reasoning-token allowance in the "
                         "estimate (default: effort-based).")
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                    help="Parallel LLM calls in the phase-1 fill (asyncio "
                         f"semaphore). Default {DEFAULT_CONCURRENCY}; 1 = "
                         "serial. Results are identical either way (each "
                         "(query, trial) call is independent).")
    ap.add_argument("--rebuild-pool", action="store_true",
                    help="Force-rebuild the candidate pool(s) from cache + corpus.")
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Reuse prior results (no LLM calls / no re-pay): a whole "
                         "domain's listwise.json when its config matches exactly, "
                         "AND any individual query already in the per-query cache "
                         "(so partial/larger runs resume). ON by default; pass "
                         "--no-skip-existing to force fresh recomputation.")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress the live per-LLM-call progress lines.")
    args = ap.parse_args()
    verbose = not args.quiet
    if args.trials < 1:
        raise SystemExit("--trials must be >= 1.")

    global POOL_SOURCE
    POOL_SOURCE = args.pool_source

    listwise_dir = RESULTS_DIR / "listwise"
    model_id = (args.model.split("/", 1)[-1]
                if args.model.startswith("openai/") else args.model)

    # Resolve prices: explicit CLI > MODEL_PRICES registry > hard stop. Both
    # the pre-flight estimate and the reported actual_usd use these, so an
    # unpriced model must not fall back to another model's rates.
    if args.price_in is None or args.price_out is None:
        if model_id not in MODEL_PRICES:
            raise SystemExit(
                f"No known pricing for model {model_id!r}. Pass --price-in and "
                "--price-out ($/1M tokens), or add the model to MODEL_PRICES in "
                "retrieval/listwise_rerank.py."
            )
        reg_in, reg_out = MODEL_PRICES[model_id]
        args.price_in = args.price_in if args.price_in is not None else reg_in
        args.price_out = args.price_out if args.price_out is not None else reg_out
    # Per-config, per-model output dir so different pool sources / sizes /
    # models / efforts never clobber each other. The run identity —
    # (pool-slug, first_k, pool_k, model, effort) — matches the cache filename
    # key exactly. (pools/ and cache/ stay shared under listwise_dir — their
    # filenames already encode the config.)
    run_dir = (listwise_dir / "runs" /
               f"{POOL_SOURCE.replace('_', '-')}__first{args.first_stage_k}"
               f"__top{args.pool_k}" /
               f"{model_id}__{args.reasoning_effort}")

    # --- One-time pool materialization for all 5 subsets -------------------- #
    if args.build_all_pools:
        build_all_pools(listwise_dir, args.first_stage_k, args.pool_k,
                        args.rebuild_pool)
        return

    domains = BRIGHT_SUBSETS if args.all_domains else [args.domain]
    # The effort-keyed allowances were measured on a 100-doc pool; reasoning
    # spend scales with pool size, so scale the estimate by pool_k/100.
    base_allowance = REASONING_ALLOWANCE.get(args.reasoning_effort, 800)
    if args.reasoning_tokens_per_call is not None:
        reasoning_per_call = args.reasoning_tokens_per_call
    elif args.reasoning_effort == "none":
        # Reasoning is off: no hidden tokens, no floor.
        reasoning_per_call = 0
    else:
        reasoning_per_call = max(REASONING_ALLOWANCE_FLOOR,
                                 round(base_allowance * args.pool_k
                                       / REASONING_ALLOWANCE_POOL_K))
    enc = get_encoder(args.model)

    # --- Phase 1: load pools, sample, per-query cache, pre-flight estimate --- #
    preps = [prepare_domain(listwise_dir, d, args, enc, reasoning_per_call, model_id)
             for d in domains]
    for p in preps:  # reuse a whole prior matching result instead of re-paying?
        p["reuse"] = matching_existing(run_dir, p["domain"], p, args, model_id)
    print(f"\n=== PRE-FLIGHT (single-pass listwise · {POOL_SOURCE} top-"
          f"{args.pool_k} pool · {args.trials} trials/query) ===")
    print(f"  {'subset':<14} {'n':>3} {'pool R@1 (pool / best sgl / lift)':<34} {'est $':>8}")
    total = 0.0
    for p in preps:
        r1 = p["meta"]["pool_r1"]
        best = max(r1[x] for x in PROVIDERS)
        if p["reuse"]:
            note = f"reuse domain result (${p['reuse']['cost']['actual_usd']:.3f})"
        else:
            cached = f" ({p['n_cached']} cached)" if p["n_cached"] else ""
            note = f"${p['est']['estimated_usd']:>7.3f}{cached}"
            total += p["est"]["estimated_usd"]
        print(f"  {p['domain']:<14} {p['n']:>3} "
              f"{r1[POOL_SOURCE]:.3f} / {best:.3f} / {r1[POOL_SOURCE]-best:+.3f}"
              f"{'':<12} {note}")
    n_calls = sum(p["est"]["n_calls"] for p in preps if not p["reuse"])
    n_reuse = sum(1 for p in preps if p["reuse"])
    reuse_note = f"  ({n_reuse} domain(s) reused)" if n_reuse else ""
    print(f"  {'TOTAL':<14} {'':>3} {n_calls} new calls{'':<23} ${total:>7.3f}{reuse_note}")

    if args.dry_run:
        print("\n[--dry-run] No LLM calls made. Re-run without --dry-run to "
              "execute (requires OPENAI_API_KEY).")
        return

    # Domains that still need fresh LLM calls (not fully reused / cached).
    needs_llm = [p for p in preps
                 if not p["reuse"] and p["n_cached"] < p["n_calls_total"]]
    if needs_llm and not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set (required for the real run).")

    # --- OpenAI client (raw async SDK; structured outputs) ------------------ #
    cfg = None
    if needs_llm:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=5)
        cfg = LLMConfig(client=client, model=model_id,
                        reasoning_effort=args.reasoning_effort,
                        max_completion_tokens=MAX_COMPLETION_TOKENS)
        write_prompts(listwise_dir)

    summaries = []
    for p in preps:
        if p["reuse"]:
            print(f"  [{p['domain']}] reused existing result "
                  f"(${p['reuse']['cost']['actual_usd']:.3f}, no API calls).")
            summaries.append(summary_from_payload(p["reuse"]))
            continue
        # Phase 1: concurrently fill the per-query cache (only missing calls).
        fresh_keys: set[tuple[str, int]] = set()
        if p["n_cached"] < p["n_calls_total"]:
            assert cfg is not None  # needs_llm guaranteed the client exists
            fresh_keys = asyncio.run(fill_query_cache(p, cfg, args, verbose))
        # Phase 2: serial aggregation over the complete cache (no LLM calls).
        summaries.append(run_domain_listwise(p, args, run_dir, verbose,
                                             fresh_keys))
    if len(summaries) > 1:
        write_cross_subset_summary(summaries, run_dir, args)
    elif preps[0]["reuse"]:
        print(f"\nReused existing {run_dir / summaries[0]['domain']}/"
              "listwise.json (no API calls; nothing re-written).")
    else:
        print(f"\nWrote {run_dir / summaries[0]['domain']}/listwise.json "
              "+ listwise_table.md + rescues_table.md + listwise_plot.png "
              f"(prompts in {listwise_dir}/prompts/)")


if __name__ == "__main__":
    main()
