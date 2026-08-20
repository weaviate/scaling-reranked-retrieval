#!/usr/bin/env python3
"""Top-100 listwise reranking — Conditions A and B (GPT-5.6 Luna only).

Spec: extend the listwise experiment from a 20-doc to a 100-doc window.
  Condition A ("does the window scale?")   — Luna reranks the top-100 retained
    by Zerank (cached zerank-2 scores over the k=2000 hybrid pool), descending
    score order. Input recall ceiling: median 0.711.
  Condition B ("can listwise replace the CE?") — Luna reranks the first 100
    docs of the cached nested-prefix hybrid ordering (no cross-encoder).
    Input recall ceiling: median 0.494.

Identical model/prompt/effort to the existing top-20 runs (imported from
retrieval/listwise_rerank.py — SYSTEM_PROMPT / USER_TEMPLATE / RESPONSE_FORMAT
are shared objects, not copies); the only change is n=100 passages. 3 trials
per query per condition. Zero cross-encoder / retrieval calls — inputs come
from caches/k2000.json.

ISOLATION: all artifacts land under results/raw/listwise_top100/ and the final
report at results/listwise_top100_results.md. No existing results file is read
for writing or modified.

CLI (run in order):
  uv run python scripts/listwise_top100.py --build-pools   # $0, one-time
  uv run python scripts/listwise_top100.py --dry-run       # $0, token estimate
  uv run python scripts/listwise_top100.py --pilot         # 10 q/condition, 1 trial
  uv run python scripts/listwise_top100.py --run           # full 3-trial runs (gated on pilot)
  uv run python scripts/listwise_top100.py --analyze       # $0, writes the report

ENV: OPENAI_API_KEY (or a line `OPENAI_API_KEY=...` in the repo-root .env) for
--pilot / --run only.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import statistics
import time
from pathlib import Path

from query_agent_benchmarking.internal.adapters.dataset import (
    in_memory_dataset_loader,
)

from src.adapters.cache import ScoreCache, validate_cache_for_use
from src.config import (
    BRIGHT_SUBSETS,
    CACHE_K,
    DATASETS,
    MODEL_OVERRIDES,
    REPO_ROOT,
    RESULTS_DIR,
    get_results_dir,
)
from src.domain.metrics import metric

# The prompt/schema are IMPORTED from the existing harness so they cannot
# drift from the top-20 runs (spec: identical prompt, only {n}=100 changes).
from src.application.experiments.listwise_rerank import (  # noqa: E402  (triggers qab.setup())
    MAX_COMPLETION_TOKENS,
    RESPONSE_FORMAT,
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    build_user_prompt,
    count_message_tokens,
    extract_ints,
    get_encoder,
    load_corpus_text_map,
)

# --------------------------------------------------------------------------- #
# Configuration (fixed by the spec)                                            #
# --------------------------------------------------------------------------- #

MODEL = "gpt-5.6-luna"
EFFORT = "none"
TRIALS = 3
WINDOW = 100
SEED = 42
DEFAULT_CONCURRENCY = 8
# Luna list price effective 2026-07-30 ($/1M input, output) — per the spec;
# supersedes the older 1.00/6.00 entry in retrieval/listwise_rerank.MODEL_PRICES.
PRICE_IN, PRICE_OUT = 0.20, 1.20
SPEC_INPUT_TOKENS_PER_CALL = 50_400  # original spec estimate (kept for reporting)
PILOT_QUERIES_PER_SUBSET = 2         # 2 x 5 subsets = 10 queries per condition
# Amendment 1 gates: token gate = expected MEAN band (actual BRIGHT docs are
# ~110-180 tokens, not the d=500 modeling cap, so the original 50,400 estimate
# was ~2x high — the -58% pilot deviation is explained, no re-run needed).
# Malformed-rate gate SUSPENDED pending the severity analysis (--severity);
# the decision rule there is head-intact@20 >= 90%.
PILOT_TOKEN_BAND = (15_000, 30_000)
PILOT_MAX_MALFORMED_RATE = 0.20      # original spec value; report-only now
PILOT_MAX_TOKEN_DEVIATION = 0.25     # original spec value; report-only now
HEAD_INTACT_DECISION_RATE = 0.90     # amendment 1 decision rule (head-intact@20)
# Amendment-1 resolution: if FULL-RUN head-intact@20 drops below this, note it
# prominently at the top of the results file (do not stop the run).
FULL_RUN_HEAD_INTACT_WARN = 0.95
# Invariant: input-pool R@100 cross-subset medians pinned by the spec. analyze()
# refuses to write results if the measured baseline medians do not match.
PINNED_INPUT_R100 = {"A": 0.711, "B": 0.494}

CONDITIONS = ("A", "B")
COND_SLUG = {"A": "zerank_top100", "B": "hybrid_top100"}
COND_ORDERING = {
    "A": ("top-100 documents by cached zerank-2 score over the k=2000 hybrid "
          "pool, in DESCENDING ZERANK SCORE order (not shuffled/re-sorted)"),
    "B": ("first 100 documents of the cached nested-prefix hybrid ordering "
          "(k=2000 retrieval), in HYBRID FUSED-SCORE order (not shuffled/re-sorted)"),
}

RAW_DIR = RESULTS_DIR / "raw" / "listwise_top100"
POOLS_DIR = RAW_DIR / "pools"
CACHE_DIR = RAW_DIR / "cache"
PILOT_REPORT = RAW_DIR / "pilot_report.json"
SEVERITY_REPORT = RAW_DIR / "severity_report.json"
ANALYSIS_JSON = RAW_DIR / "analysis.json"
RESULTS_MD = RESULTS_DIR / "listwise_top100_results.md"

METRICS = ("nDCG_at_10", "recall_at_1", "recall_at_20", "recall_at_100")
MET_LABEL = {"nDCG_at_10": "nDCG@10", "recall_at_1": "S@1",
             "recall_at_20": "R@20", "recall_at_100": "R@100"}

PROMPT_HASH = hashlib.sha256(
    (SYSTEM_PROMPT + "\x00" + USER_TEMPLATE + "\x00"
     + json.dumps(RESPONSE_FORMAT, sort_keys=True)).encode()
).hexdigest()

# Reference rows for the report (values verified against the cached runs files
# results/bright_*/runs_rk100/k2000_from_k2000.json — cross-subset medians).
REFERENCE_ROWS = [
    ("Zerank top-20 input ordering (cached)", {"nDCG_at_10": 0.451,
                                               "recall_at_1": 0.376,
                                               "recall_at_20": 0.597}),
    ("Luna over Zerank top-20 (cached)",      {"nDCG_at_10": 0.515,
                                               "recall_at_1": 0.492}),
    ("Hybrid ordering, no rerank (cached)",   {"nDCG_at_10": 0.174,
                                               "recall_at_20": 0.256}),
]


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def std(xs):
    return statistics.stdev(xs) if len(xs) >= 2 else 0.0


def pool_hash(doc_ids: list[str]) -> str:
    return hashlib.md5("\x00".join(doc_ids).encode()).hexdigest()[:16]


def load_api_key() -> "str | None":
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    envf = REPO_ROOT / ".env"
    if envf.exists():
        for line in envf.read_text().splitlines():
            line = line.strip()
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip().strip("'\"")
    return None


# --------------------------------------------------------------------------- #
# Permutation classification + repair (spec §"Permutation validation")         #
# --------------------------------------------------------------------------- #


def classify_and_repair(raw_ints: list[int], n: int) -> dict:
    """Classify a returned ranking and mechanically repair it (spec policy):
    1. keep the FIRST occurrence of each in-range ID (drop dups/out-of-range);
    2. append all missing IDs at the end IN INPUT ORDER.
    Failure flags (a response can carry several): duplicate, missing,
    out_of_range, truncated, unparseable. valid = exact permutation of 1..n."""
    if not raw_ints:
        return {"repaired": list(range(1, n + 1)), "valid": False,
                "flags": ["unparseable"], "n_appended": n,
                "raw_len": 0}
    flags: list[str] = []
    in_range = [x for x in raw_ints if 1 <= x <= n]
    if len(in_range) < len(raw_ints):
        flags.append("out_of_range")
    seen: set[int] = set()
    order: list[int] = []
    for x in in_range:
        if x in seen:
            if "duplicate" not in flags:
                flags.append("duplicate")
        else:
            seen.add(x)
            order.append(x)
    missing = [i for i in range(1, n + 1) if i not in seen]
    if missing:
        flags.append("truncated" if len(raw_ints) < n else "missing")
    repaired = order + missing
    return {"repaired": repaired, "valid": not flags, "flags": flags,
            "n_appended": len(missing), "raw_len": len(raw_ints)}


def severity_of(raw_ints: list[int], n: int) -> dict:
    """Amendment-1 severity metrics for one response (pre-repair):
    - n_valid_unique: count of valid unique in-range IDs returned.
    - head_intact_20 / head_intact_10: n_valid_unique >= 20 / >= 10.
    - first_repair_rank: output position of the first repair-affected slot —
      the slot where the first dropped duplicate/out-of-range element would
      have gone (kept-so-far + 1), else the first appended slot
      (n_valid_unique + 1); None for a valid permutation.
    """
    seen: set[int] = set()
    kept = 0
    first_affected = None
    for x in raw_ints:
        if 1 <= x <= n and x not in seen:
            seen.add(x)
            kept += 1
        elif first_affected is None:
            first_affected = kept + 1
    if first_affected is None and kept < n:
        first_affected = kept + 1  # first appended slot
    return {"n_valid_unique": kept,
            "head_intact_20": kept >= 20,
            "head_intact_10": kept >= 10,
            "first_repair_rank": first_affected,
            "raw_len": len(raw_ints)}


# --------------------------------------------------------------------------- #
# Pool construction (zero API calls)                                           #
# --------------------------------------------------------------------------- #


def load_cache(domain: str) -> ScoreCache:
    cfg = DATASETS[domain]
    path = get_results_dir(domain) / "caches" / f"k{CACHE_K}.json"
    if not path.exists():
        raise SystemExit(f"No cache at {path}")
    cache = ScoreCache.load(path)
    validate_cache_for_use(cache, needed_retrieved_k=CACHE_K,
                           expected_model_overrides=MODEL_OVERRIDES,
                           expected_dataset=cfg.qab_name,
                           expected_collection=cfg.collection)
    return cache


def input_ids_for(entry: dict, cond: str) -> "list[str] | None":
    """The condition's 100-doc input list, in the pipeline's own order."""
    pool = entry["hybrid_order"][:CACHE_K]
    if cond == "A":
        sc = entry.get("zerank_scores") or {}
        if not sc:
            return None  # zerank never scored this query (documented drops)
        scores = {d: sc[d] for d in pool if d in sc}
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [d for d, _ in ranked[:WINDOW]]
    return pool[:WINDOW]


def pool_path(domain: str, cond: str) -> Path:
    return POOLS_DIR / f"{domain}__{COND_SLUG[cond]}.json"


def build_pools() -> None:
    POOLS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{'subset':<14} {'cond':<4} {'n':>4} {'skipped':>7} "
          f"{'input R@100':>11} {'input R@20':>10} {'input S@1':>9}")
    for domain in BRIGHT_SUBSETS:
        cache = load_cache(domain)
        cfg = DATASETS[domain]
        _, queries = in_memory_dataset_loader(cfg.qab_name, queries_only=True)
        gold_by_text = {q.question: sorted({str(d) for d in q.dataset_ids})
                        for q in queries}
        qid_by_text = {q.question: (q.query_id or q.question[:64]) for q in queries}
        id2text = load_corpus_text_map(cfg.qab_name)
        for cond in CONDITIONS:
            qmap: dict[str, dict] = {}
            doc_texts: dict[str, str] = {}
            skipped: list[str] = []
            for text in sorted(cache.queries.keys()):
                gold = gold_by_text.get(text)
                if gold is None:
                    continue  # no gold labels for this cache query
                ids = input_ids_for(cache.queries[text], cond)
                if ids is None:
                    skipped.append(qid_by_text.get(text, text[:32]))
                    continue
                qmap[text] = {"qid": str(qid_by_text[text]), "gold": gold,
                              "doc_ids": ids}
                for d in ids:
                    if d not in doc_texts:
                        doc_texts[d] = id2text[d]
            agg = {m: mean([metric(m, q["gold"], q["doc_ids"])
                            for q in qmap.values()]) for m in METRICS}
            payload = {
                "metadata": {
                    "domain": domain, "condition": cond,
                    "condition_slug": COND_SLUG[cond],
                    "input_ordering": COND_ORDERING[cond],
                    "window": WINDOW, "cache_retrieved_k": CACHE_K,
                    "model_overrides": MODEL_OVERRIDES,
                    "n_queries": len(qmap),
                    "skipped_queries_no_zerank": skipped,
                    "input_order_metrics": agg,
                },
                "doc_texts": doc_texts,
                "queries": qmap,
            }
            with open(pool_path(domain, cond), "w") as f:
                json.dump(payload, f)
            print(f"{domain:<14} {cond:<4} {len(qmap):>4} {len(skipped):>7} "
                  f"{agg['recall_at_100']:>11.3f} {agg['recall_at_20']:>10.3f} "
                  f"{agg['recall_at_1']:>9.3f}")
    print(f"\nPools saved under {POOLS_DIR}/ (input ordering preserved per "
          "condition; R@100 above is each condition's input recall ceiling).")


def load_pool(domain: str, cond: str) -> dict:
    p = pool_path(domain, cond)
    if not p.exists():
        raise SystemExit(f"Missing pool {p}; run --build-pools first.")
    with open(p) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Response cache (append-only JSONL; resumable)                                #
# --------------------------------------------------------------------------- #


def cache_file(domain: str, cond: str) -> Path:
    return CACHE_DIR / f"{domain}__{COND_SLUG[cond]}__{MODEL}__{EFFORT}.jsonl"


def load_response_cache(domain: str, cond: str) -> dict:
    out: dict[tuple[str, int], dict] = {}
    p = cache_file(domain, cond)
    if not p.exists():
        return out
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if "qid" in e and "ranking_repaired" in e:
                out[(str(e["qid"]), int(e["trial"]))] = e
    return out


def append_response_cache(domain: str, cond: str, entry: dict) -> None:
    p = cache_file(domain, cond)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(entry) + "\n")


# --------------------------------------------------------------------------- #
# LLM call + concurrent fill                                                   #
# --------------------------------------------------------------------------- #


async def call_llm(client, user: str) -> dict:
    t0 = time.perf_counter()
    resp = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": user}],
        response_format=RESPONSE_FORMAT,
        reasoning_effort=EFFORT,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
    )
    dt = time.perf_counter() - t0
    u = resp.usage
    details = getattr(u, "completion_tokens_details", None)
    return {
        "text": resp.choices[0].message.content or "",
        "finish_reason": resp.choices[0].finish_reason,
        "model_snapshot": resp.model,
        "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
        "reasoning_tokens": (getattr(details, "reasoning_tokens", 0) or 0)
        if details is not None else 0,
        "latency_s": round(dt, 3),
    }


def build_jobs(scope: dict, trials: int) -> list[dict]:
    """scope: {(domain, cond): [query_text, ...]}. Returns uncached jobs."""
    jobs = []
    for (domain, cond), texts in scope.items():
        pool = load_pool(domain, cond)
        rcache = load_response_cache(domain, cond)
        for text in texts:
            q = pool["queries"][text]
            ph = pool_hash(q["doc_ids"])
            for tr in range(trials):
                e = rcache.get((q["qid"], tr))
                if e is not None and e.get("pool_hash") == ph:
                    continue
                jobs.append({"domain": domain, "cond": cond, "text": text,
                             "qid": q["qid"], "trial": tr,
                             "doc_ids": q["doc_ids"],
                             "texts": [pool["doc_texts"][d] for d in q["doc_ids"]],
                             "pool_hash": ph})
    return jobs


async def fill(jobs: list[dict], concurrency: int, verbose: bool) -> list[dict]:
    """Run all jobs concurrently; append each to its JSONL as it lands.
    Returns the list of failures (exceptions after SDK retries)."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=load_api_key(), max_retries=5)
    sem = asyncio.Semaphore(max(1, concurrency))
    lock = asyncio.Lock()
    done = 0
    spent = 0.0
    failures: list[dict] = []

    async def one(j: dict):
        nonlocal done, spent
        n = len(j["doc_ids"])
        user = build_user_prompt(j["text"], j["texts"])
        try:
            async with sem:
                r = await call_llm(client, user)
        except Exception as exc:  # terminal after SDK retries; keep going
            async with lock:
                failures.append({"qid": j["qid"], "trial": j["trial"],
                                 "domain": j["domain"], "cond": j["cond"],
                                 "error": repr(exc)})
                done += 1
                print(f"  [{done}/{len(jobs)}] {j['domain']}/{j['cond']} "
                      f"q{j['qid']} t{j['trial']} FAILED: {exc!r}", flush=True)
            return
        raw = extract_ints(r["text"])
        cls = classify_and_repair(raw, n)
        entry = {
            "qid": j["qid"], "trial": j["trial"], "condition": j["cond"],
            "domain": j["domain"], "pool_hash": j["pool_hash"], "n": n,
            "ranking_raw": raw, "ranking_repaired": cls["repaired"],
            "valid": cls["valid"], "flags": cls["flags"],
            "n_appended": cls["n_appended"],
            "finish_reason": r["finish_reason"],
            "model_snapshot": r["model_snapshot"],
            "prompt_tokens": r["prompt_tokens"],
            "completion_tokens": r["completion_tokens"],
            "reasoning_tokens": r["reasoning_tokens"],
            "latency_s": r["latency_s"],
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        async with lock:
            append_response_cache(j["domain"], j["cond"], entry)
            done += 1
            spent += (r["prompt_tokens"] * PRICE_IN
                      + r["completion_tokens"] * PRICE_OUT) / 1e6
            if verbose:
                tag = "" if cls["valid"] else f" · MALFORMED {cls['flags']}"
                print(f"  [{done}/{len(jobs)}] {j['domain']}/{j['cond']} "
                      f"q{j['qid']} t{j['trial']} · {r['latency_s']:5.1f}s · "
                      f"in {r['prompt_tokens']:>6,} · ${spent:.2f} so far{tag}",
                      flush=True)

    await asyncio.gather(*(one(j) for j in jobs))
    return failures


# --------------------------------------------------------------------------- #
# Dry-run token estimate                                                       #
# --------------------------------------------------------------------------- #


def full_scope(conds=CONDITIONS) -> dict:
    return {(d, c): sorted(load_pool(d, c)["queries"].keys())
            for d in BRIGHT_SUBSETS for c in conds}


def dry_run() -> None:
    enc = get_encoder(MODEL)
    schema_tok = len(enc.encode(json.dumps(RESPONSE_FORMAT)))
    out_per_call = len(enc.encode(str(list(range(1, WINDOW + 1)))))
    grand_in = 0
    n_calls = 0
    print(f"{'subset':<14} {'cond':<4} {'n':>4} {'mean in-tok/call':>16} "
          f"{'min':>8} {'max':>8} {'vs spec 50,400':>14}")
    for domain in BRIGHT_SUBSETS:
        for cond in CONDITIONS:
            pool = load_pool(domain, cond)
            counts = []
            for text, q in pool["queries"].items():
                user = build_user_prompt(
                    text, [pool["doc_texts"][d] for d in q["doc_ids"]])
                counts.append(count_message_tokens(enc, SYSTEM_PROMPT, user)
                              + schema_tok)
            m = mean(counts)
            dev = (m - SPEC_INPUT_TOKENS_PER_CALL) / SPEC_INPUT_TOKENS_PER_CALL
            print(f"{domain:<14} {cond:<4} {len(counts):>4} {m:>16,.0f} "
                  f"{min(counts):>8,} {max(counts):>8,} {dev:>+13.0%}")
            grand_in += sum(counts) * TRIALS
            n_calls += len(counts) * TRIALS
    est_out = out_per_call * n_calls
    usd = (grand_in * PRICE_IN + est_out * PRICE_OUT) / 1e6
    print(f"\nFull experiment: {n_calls:,} calls ({TRIALS} trials x both "
          f"conditions) · ~{grand_in/1e6:.1f}M input tok · ~{est_out/1e6:.2f}M "
          f"visible output tok\nEstimated cost at ${PRICE_IN:.2f}/"
          f"${PRICE_OUT:.2f} per 1M: ${usd:.2f}")


# --------------------------------------------------------------------------- #
# Pilot                                                                        #
# --------------------------------------------------------------------------- #


def pilot_scope() -> dict:
    scope = {}
    for domain in BRIGHT_SUBSETS:
        for cond in CONDITIONS:
            qlist = sorted(load_pool(domain, cond)["queries"].keys())
            k = min(PILOT_QUERIES_PER_SUBSET, len(qlist))
            scope[(domain, cond)] = random.Random(SEED).sample(qlist, k)
    return scope


def run_pilot(concurrency: int) -> None:
    if not load_api_key():
        raise SystemExit("OPENAI_API_KEY is not set (env or repo-root .env).")
    scope = pilot_scope()
    jobs = build_jobs(scope, trials=1)
    n_total = sum(len(v) for v in scope.values())
    print(f"Pilot: {n_total} (query, condition) pairs, 1 trial; "
          f"{len(jobs)} uncached calls.")
    t0 = time.time()
    failures = asyncio.run(fill(jobs, concurrency, verbose=True))
    wall = time.time() - t0

    # Collect the pilot entries (trial 0 of the sampled queries).
    entries = []
    for (domain, cond), texts in scope.items():
        pool = load_pool(domain, cond)
        rcache = load_response_cache(domain, cond)
        for text in texts:
            e = rcache.get((pool["queries"][text]["qid"], 0))
            if e is not None:
                entries.append(e)
    n_ok = len(entries)
    malformed = [e for e in entries if not e["valid"]]
    in_toks = [e["prompt_tokens"] for e in entries]
    lats = sorted(e["latency_s"] for e in entries)
    mean_in = mean(in_toks)
    dev = ((mean_in - SPEC_INPUT_TOKENS_PER_CALL) / SPEC_INPUT_TOKENS_PER_CALL
           if in_toks else 0.0)
    mal_rate = len(malformed) / n_ok if n_ok else 1.0
    # Amendment-1 gates: token MEAN band + no API errors. Malformed rate is
    # report-only (suspended); the go/no-go on it is --severity's
    # head-intact@20 >= 90% decision rule.
    gate_ok = (not failures
               and PILOT_TOKEN_BAND[0] <= mean_in <= PILOT_TOKEN_BAND[1])
    report = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_planned": n_total, "n_ok": n_ok,
        "api_errors": failures,
        "malformed_count": len(malformed),
        "malformed_rate": round(mal_rate, 4),
        "malformed_detail": [{"qid": e["qid"], "condition": e["condition"],
                              "domain": e["domain"], "flags": e["flags"],
                              "n_appended": e["n_appended"]} for e in malformed],
        "input_tokens": {"mean": round(mean_in, 1),
                         "min": min(in_toks) if in_toks else None,
                         "max": max(in_toks) if in_toks else None,
                         "per_call": {f"{e['domain']}/{e['condition']}/q{e['qid']}":
                                      e["prompt_tokens"] for e in entries},
                         "spec_estimate": SPEC_INPUT_TOKENS_PER_CALL,
                         "deviation_from_spec": round(dev, 4)},
        "latency_s": {"mean": round(mean(lats), 2) if lats else None,
                      "median": lats[len(lats) // 2] if lats else None,
                      "min": lats[0] if lats else None,
                      "max": lats[-1] if lats else None,
                      "wall_clock_s": round(wall, 1)},
        "model_snapshots": sorted({e["model_snapshot"] for e in entries}),
        "gate": {"token_mean_band": list(PILOT_TOKEN_BAND),
                 "malformed_gate": "suspended (amendment 1) — see --severity "
                                   "head-intact@20 decision rule",
                 "original_spec_thresholds": {
                     "max_malformed_rate": PILOT_MAX_MALFORMED_RATE,
                     "max_token_deviation": PILOT_MAX_TOKEN_DEVIATION},
                 "passed": gate_ok},
    }
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PILOT_REPORT.write_text(json.dumps(report, indent=2))
    print(f"\n=== PILOT REPORT ({PILOT_REPORT}) ===")
    print(f"  calls ok {n_ok}/{n_total}, API errors {len(failures)}")
    print(f"  malformed {len(malformed)}/{n_ok} = {mal_rate:.1%} "
          f"(gate <= {PILOT_MAX_MALFORMED_RATE:.0%})")
    print(f"  input tokens mean {mean_in:,.0f} vs spec "
          f"{SPEC_INPUT_TOKENS_PER_CALL:,} ({dev:+.0%}; gate ±"
          f"{PILOT_MAX_TOKEN_DEVIATION:.0%})")
    if lats:
        print(f"  latency median {lats[len(lats)//2]:.1f}s "
              f"(min {lats[0]:.1f} / max {lats[-1]:.1f}), wall {wall:.0f}s")
    print(f"  GATE: {'PASSED — proceed with --run' if gate_ok else 'FAILED — STOP AND REPORT (spec)'}")


# --------------------------------------------------------------------------- #
# Severity analysis (amendment 1; zero API calls)                              #
# --------------------------------------------------------------------------- #


def run_severity() -> None:
    """Pilot severity analysis over the 20 cached pilot responses (no new API
    calls). Reports pre-repair validity counts, head-intact@20/@10, the first
    repair-affected rank per malformed response, and truncation lengths; then
    applies the amendment-1 decision rule (head-intact@20 >= 90% -> full run
    proceeds unchanged). Report-only: no design change is made here."""
    scope = pilot_scope()
    rows: list[dict] = []
    for (domain, cond), texts in sorted(scope.items()):
        pool = load_pool(domain, cond)
        rcache = load_response_cache(domain, cond)
        for text in texts:
            q = pool["queries"][text]
            e = rcache.get((q["qid"], 0))
            if e is None:
                raise SystemExit(f"Missing pilot entry {domain}/{cond} "
                                 f"q{q['qid']} trial 0; run --pilot first.")
            cls = classify_and_repair(e["ranking_raw"], e["n"])
            sev = severity_of(e["ranking_raw"], e["n"])
            rows.append({"domain": domain, "condition": cond, "qid": q["qid"],
                         "valid": cls["valid"], "flags": cls["flags"],
                         "n_appended": cls["n_appended"], **sev})
    n = len(rows)
    head20 = sum(r["head_intact_20"] for r in rows)
    head10 = sum(r["head_intact_10"] for r in rows)
    malformed = [r for r in rows if not r["valid"]]
    trunc_lens = sorted(r["raw_len"] for r in rows if "truncated" in r["flags"])
    first_ranks = sorted(r["first_repair_rank"] for r in malformed
                         if r["first_repair_rank"] is not None)
    decision = ("PROCEED" if n and head20 / n >= HEAD_INTACT_DECISION_RATE
                else "STOP")
    report = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_responses": n,
        "per_response": rows,
        "head_intact_20_count": head20,
        "head_intact_20_rate": round(head20 / n, 4) if n else None,
        "head_intact_10_count": head10,
        "head_intact_10_rate": round(head10 / n, 4) if n else None,
        "malformed_count": len(malformed),
        "first_repair_rank": {
            "values": first_ranks,
            "min": first_ranks[0] if first_ranks else None,
            "median": first_ranks[len(first_ranks) // 2] if first_ranks else None,
        },
        "truncation_lengths": trunc_lens,
        "decision_rule": {
            "threshold_head_intact_20": HEAD_INTACT_DECISION_RATE,
            "decision": decision,
            "note": ("PROCEED = full run proceeds unchanged (malformed + "
                     "head-intact rates reported per the original spec). "
                     "STOP = report; design amendment is the author's call "
                     "(candidates: top-20-of-100 output, or sliding-window "
                     "n=20 passes)."),
        },
    }
    SEVERITY_REPORT.write_text(json.dumps(report, indent=2))

    # Refresh the pilot report's gate to the amendment-1 rules so --run's
    # gate check reflects the amended spec.
    if PILOT_REPORT.exists():
        rep = json.loads(PILOT_REPORT.read_text())
        mean_in = rep["input_tokens"]["mean"]
        token_ok = PILOT_TOKEN_BAND[0] <= mean_in <= PILOT_TOKEN_BAND[1]
        rep["gate"] = {
            "token_mean_band": list(PILOT_TOKEN_BAND),
            "token_mean": mean_in,
            "token_ok": token_ok,
            "malformed_gate": "suspended (amendment 1) — decision via "
                              "severity_report.json head-intact@20",
            "severity_decision": decision,
            "passed": bool(token_ok and not rep.get("api_errors")
                           and decision == "PROCEED"),
            "original_spec_thresholds": {
                "max_malformed_rate": PILOT_MAX_MALFORMED_RATE,
                "max_token_deviation": PILOT_MAX_TOKEN_DEVIATION},
        }
        PILOT_REPORT.write_text(json.dumps(rep, indent=2))

    print(f"=== SEVERITY ANALYSIS ({SEVERITY_REPORT}) ===")
    print(f"{'resp':<28} {'valid':>5} {'n_valid':>7} {'h@20':>5} {'h@10':>5} "
          f"{'1st-repair':>10} {'raw_len':>7} flags")
    for r in rows:
        tag = f"{r['domain']}/{r['condition']}/q{r['qid']}"
        fr = r["first_repair_rank"]
        print(f"{tag:<28} {str(r['valid']):>5} {r['n_valid_unique']:>7} "
              f"{'y' if r['head_intact_20'] else 'N':>5} "
              f"{'y' if r['head_intact_10'] else 'N':>5} "
              f"{fr if fr is not None else '—':>10} {r['raw_len']:>7} "
              f"{','.join(r['flags']) or '—'}")
    print(f"\n  head-intact@20: {head20}/{n} = {head20/n:.0%} "
          f"(threshold {HEAD_INTACT_DECISION_RATE:.0%}) · head-intact@10: "
          f"{head10}/{n} = {head10/n:.0%}")
    print(f"  malformed {len(malformed)}/{n}; first repair-affected rank: "
          f"min {report['first_repair_rank']['min']}, median "
          f"{report['first_repair_rank']['median']}")
    print(f"  truncation lengths: {trunc_lens or '—'}")
    print(f"  DECISION: {decision}"
          + (" — full run proceeds unchanged." if decision == "PROCEED" else
             " — stop and report; design amendment is the author's call."))


# --------------------------------------------------------------------------- #
# Full run                                                                     #
# --------------------------------------------------------------------------- #


def run_full(concurrency: int, force: bool) -> None:
    if not load_api_key():
        raise SystemExit("OPENAI_API_KEY is not set (env or repo-root .env).")
    if not force:
        if not PILOT_REPORT.exists():
            raise SystemExit("No pilot report; run --pilot first (or --force).")
        if not SEVERITY_REPORT.exists():
            raise SystemExit("No severity report; run --severity first "
                             "(amendment 1) or override with --force.")
        rep = json.loads(PILOT_REPORT.read_text())
        if not rep.get("gate", {}).get("passed"):
            raise SystemExit("Pilot gate FAILED (amendment-1 rules: token "
                             "band / API errors / severity decision) — stop "
                             "and report. Override with --force.")
    scope = full_scope()
    jobs = build_jobs(scope, trials=TRIALS)
    n_total = sum(len(v) for v in scope.values()) * TRIALS
    est_in = SPEC_INPUT_TOKENS_PER_CALL
    if PILOT_REPORT.exists():
        est_in = json.loads(PILOT_REPORT.read_text())["input_tokens"]["mean"]
    usd = len(jobs) * (est_in * PRICE_IN + 450 * PRICE_OUT) / 1e6
    print(f"Full run: {n_total} (query, condition, trial) calls total; "
          f"{len(jobs)} uncached (~${usd:.2f} at ~{est_in:,.0f} in-tok/call).")
    t0 = time.time()
    failures = asyncio.run(fill(jobs, concurrency, verbose=True))
    wall = time.time() - t0
    print(f"\nDone: {len(jobs) - len(failures)}/{len(jobs)} calls in "
          f"{wall:.0f}s wall.")
    if failures:
        raise SystemExit(f"{len(failures)} calls failed after SDK retries — "
                         "completed calls are cached; RE-RUN --run to pay only "
                         "for the missing ones.")
    print("All calls cached. Next: uv run python scripts/listwise_top100.py --analyze")


# --------------------------------------------------------------------------- #
# Analysis + report                                                            #
# --------------------------------------------------------------------------- #


def analyze() -> None:
    conds: dict[str, dict] = {}
    all_snapshots: set[str] = set()
    ts_seen: list[str] = []
    sev_rows: list[dict] = []  # per-response severity flags (full run)
    for cond in CONDITIONS:
        per_subset: dict[str, dict] = {}
        for domain in BRIGHT_SUBSETS:
            pool = load_pool(domain, cond)
            rcache = load_response_cache(domain, cond)
            qmap = pool["queries"]
            texts = sorted(qmap.keys())
            n = len(texts)

            base_metrics = {m: mean([metric(m, qmap[t]["gold"], qmap[t]["doc_ids"])
                                     for t in texts]) for m in METRICS}
            trial_metrics = {m: [] for m in METRICS}
            promoted_tr, demoted_tr, net_tr = [], [], []
            mal_by_trial = []
            flag_counts: dict[str, int] = {}
            appended_total = 0
            tok_in = tok_out = tok_reason = 0
            lats = []
            r100_max_diff = 0.0
            head20 = head10 = 0
            n_valid_sum = 0
            first_ranks: list[int] = []
            trunc_lens: list[int] = []

            for tr in range(TRIALS):
                per_q = {m: [] for m in METRICS}
                mal = 0
                promoted = demoted = 0
                net = 0.0
                for t in texts:
                    q = qmap[t]
                    e = rcache.get((q["qid"], tr))
                    if e is None:
                        raise SystemExit(f"Missing cached call {domain}/{cond} "
                                         f"q{q['qid']} trial {tr}; complete "
                                         "--run first.")
                    if e.get("pool_hash") != pool_hash(q["doc_ids"]):
                        raise SystemExit(f"Pool mismatch for {domain}/{cond} "
                                         f"q{q['qid']} trial {tr} — pool was "
                                         "rebuilt after collection.")
                    # Re-classify from the raw ranking (source of truth).
                    cls = classify_and_repair(e["ranking_raw"], e["n"])
                    if cls["repaired"] != e["ranking_repaired"]:
                        raise SystemExit(f"Repair mismatch {domain}/{cond} "
                                         f"q{q['qid']} t{tr}")
                    if not cls["valid"]:
                        mal += 1
                        for fl in cls["flags"]:
                            flag_counts[fl] = flag_counts.get(fl, 0) + 1
                        appended_total += cls["n_appended"]
                    sev = severity_of(e["ranking_raw"], e["n"])
                    head20 += int(sev["head_intact_20"])
                    head10 += int(sev["head_intact_10"])
                    n_valid_sum += sev["n_valid_unique"]
                    if sev["first_repair_rank"] is not None:
                        first_ranks.append(sev["first_repair_rank"])
                    if "truncated" in cls["flags"]:
                        trunc_lens.append(sev["raw_len"])
                    sev_rows.append({
                        "condition": cond, "domain": domain, "qid": q["qid"],
                        "trial": tr, "valid": cls["valid"],
                        "flags": cls["flags"],
                        "n_valid_unique": sev["n_valid_unique"],
                        "head_intact_20": sev["head_intact_20"],
                        "first_repair_rank": sev["first_repair_rank"],
                        "raw_len": sev["raw_len"],
                    })
                    ranked = [q["doc_ids"][i - 1] for i in cls["repaired"]]
                    gold = q["gold"]
                    for m in METRICS:
                        per_q[m].append(metric(m, gold, ranked))
                    # Promotion decomposition (input rank vs output rank).
                    in_rank = {d: i for i, d in enumerate(q["doc_ids"], 1)}
                    out_rank = {d: i for i, d in enumerate(ranked, 1)}
                    p_q = sum(1 for d in gold if d in in_rank
                              and in_rank[d] > 20 and out_rank[d] <= 20)
                    d_q = sum(1 for d in gold if d in in_rank
                              and in_rank[d] <= 20 and out_rank[d] > 20)
                    promoted += p_q
                    demoted += d_q
                    net += (p_q - d_q) / len(gold)
                    tok_in += e["prompt_tokens"]
                    tok_out += e["completion_tokens"]
                    tok_reason += e.get("reasoning_tokens", 0)
                    lats.append(e["latency_s"])
                    all_snapshots.add(e["model_snapshot"])
                    if e.get("ts"):
                        ts_seen.append(e["ts"])
                for m in METRICS:
                    trial_metrics[m].append(mean(per_q[m]))
                r100_max_diff = max(r100_max_diff,
                                    abs(trial_metrics["recall_at_100"][tr]
                                        - base_metrics["recall_at_100"]))
                mal_by_trial.append(mal)
                promoted_tr.append(promoted)
                demoted_tr.append(demoted)
                net_tr.append(net / n)

            per_subset[domain] = {
                "n": n,
                "baseline": base_metrics,
                "trials": trial_metrics,
                "mean": {m: mean(trial_metrics[m]) for m in METRICS},
                "std": {m: std(trial_metrics[m]) for m in METRICS},
                "r100_verification_max_abs_diff": r100_max_diff,
                "malformed_by_trial": mal_by_trial,
                "malformed_total": sum(mal_by_trial),
                "calls": n * TRIALS,
                "flag_counts": flag_counts,
                "appended_ids_total": appended_total,
                "severity": {
                    "head_intact_20_rate": head20 / (n * TRIALS),
                    "head_intact_10_rate": head10 / (n * TRIALS),
                    "mean_n_valid_unique": n_valid_sum / (n * TRIALS),
                    "first_repair_rank_min": min(first_ranks) if first_ranks else None,
                    "first_repair_rank_median": (sorted(first_ranks)[len(first_ranks) // 2]
                                                 if first_ranks else None),
                    "truncation_lengths": sorted(trunc_lens),
                },
                "promotion": {
                    "promoted_mean": mean(promoted_tr), "promoted_std": std(promoted_tr),
                    "demoted_mean": mean(demoted_tr), "demoted_std": std(demoted_tr),
                    "net_r20_mean": mean(net_tr), "net_r20_std": std(net_tr),
                    "measured_r20_delta": (mean(trial_metrics["recall_at_20"])
                                           - base_metrics["recall_at_20"]),
                },
                "tokens": {"input": tok_in, "output": tok_out,
                           "reasoning": tok_reason},
                "latency_s": {"mean": round(mean(lats), 2),
                              "max": round(max(lats), 2)},
            }
        conds[cond] = per_subset

    def med(cond, key, m):
        return statistics.median(conds[cond][d][key][m] for d in BRIGHT_SUBSETS)

    # --- Invariant checks (must pass BEFORE any results are written) -------- #
    # 1. R@100 == input pool recall exactly, per subset x trial x condition.
    worst_r100 = max(conds[c][d]["r100_verification_max_abs_diff"]
                     for c in CONDITIONS for d in BRIGHT_SUBSETS)
    if worst_r100 > 1e-9:
        raise SystemExit(f"INVARIANT VIOLATED: Luna R@100 != input pool recall "
                         f"(max abs diff {worst_r100}) — something is broken; "
                         "stopping without writing results.")
    # 2. Input-pool R@100 cross-subset medians match the spec's pinned values.
    for cond, pin in PINNED_INPUT_R100.items():
        got = round(med(cond, "baseline", "recall_at_100"), 3)
        if got != pin:
            raise SystemExit(f"INVARIANT VIOLATED: condition {cond} input-pool "
                             f"R@100 median {got} != pinned {pin}; stopping "
                             "without writing results.")

    # --- Full-run severity rollups (amendment-1 resolution) ---------------- #
    def _bucket(fr: int) -> str:
        if fr <= 20:
            return "le20"
        if fr <= 50:
            return "21-50"
        if fr <= 98:
            return "51-98"
        return "ge99"

    per_ct: dict[str, dict] = {}
    for c in CONDITIONS:
        per_ct[c] = {}
        for tr in range(TRIALS):
            rs = [r for r in sev_rows if r["condition"] == c and r["trial"] == tr]
            frs = sorted(r["first_repair_rank"] for r in rs
                         if r["first_repair_rank"] is not None)
            buckets = {"le20": 0, "21-50": 0, "51-98": 0, "ge99": 0}
            for fr in frs:
                buckets[_bucket(fr)] += 1
            per_ct[c][str(tr)] = {
                "calls": len(rs),
                "malformed": sum(1 for r in rs if not r["valid"]),
                "malformed_rate": (sum(1 for r in rs if not r["valid"]) / len(rs)
                                   if rs else None),
                "head_intact_20": sum(1 for r in rs if r["head_intact_20"]),
                "head_intact_20_rate": (sum(1 for r in rs if r["head_intact_20"])
                                        / len(rs) if rs else None),
                "first_repair_rank": {
                    "min": frs[0] if frs else None,
                    "median": frs[len(frs) // 2] if frs else None,
                    "max": frs[-1] if frs else None,
                    "buckets": buckets,
                },
            }
    overall_head20_rate = (sum(1 for r in sev_rows if r["head_intact_20"])
                           / len(sev_rows) if sev_rows else None)
    sev_path = RAW_DIR / "severity_per_response.jsonl"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(sev_path, "w") as f:
        for r in sev_rows:
            f.write(json.dumps(r) + "\n")

    analysis = {
        "model": MODEL, "effort": EFFORT, "trials": TRIALS, "window": WINDOW,
        "prompt_sha256": PROMPT_HASH,
        "prices_per_1m": {"input": PRICE_IN, "output": PRICE_OUT},
        "model_snapshots": sorted(all_snapshots),
        "collection_span": [min(ts_seen), max(ts_seen)] if ts_seen else None,
        "full_run_severity": {
            "overall_head_intact_20_rate": overall_head20_rate,
            "warn_threshold": FULL_RUN_HEAD_INTACT_WARN,
            "per_condition_trial": per_ct,
            "per_response_file": str(sev_path),
        },
        "per_condition": conds,
        "medians": {cond: {"baseline": {m: med(cond, "baseline", m) for m in METRICS},
                           "mean": {m: med(cond, "mean", m) for m in METRICS},
                           "std": {m: med(cond, "std", m) for m in METRICS}}
                    for cond in CONDITIONS},
    }
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_JSON.write_text(json.dumps(analysis, indent=2))
    RESULTS_MD.write_text(render_report(analysis))
    print(f"Wrote {ANALYSIS_JSON}\nWrote {RESULTS_MD}")


def render_report(a: dict) -> str:
    L: list[str] = []
    A = L.append
    conds = a["per_condition"]
    meds = a["medians"]
    fmt = lambda v: f"{v:.3f}"

    A("# Top-100 listwise reranking — Conditions A and B (GPT-5.6 Luna)")
    A("")
    fs = a.get("full_run_severity", {})
    oh = fs.get("overall_head_intact_20_rate")
    if oh is not None and oh < FULL_RUN_HEAD_INTACT_WARN:
        A(f"> ⚠️ **FULL-RUN HEAD-INTACT@20 = {oh:.1%} — below the "
          f"{FULL_RUN_HEAD_INTACT_WARN:.0%} threshold.** The pilot's 100% did "
          "not hold at scale; see §3 for the per-condition/trial breakdown "
          "before citing head metrics.")
        A("")
    elif oh is not None:
        A(f"Full-run head-intact@20 = **{oh:.1%}** (≥ "
          f"{FULL_RUN_HEAD_INTACT_WARN:.0%}; the pilot's 100% held at scale).")
        A("")
    A("Generated by `experiments/listwise_top100.py --analyze`. All artifacts "
      "isolated under `results/raw/listwise_top100/`; no pre-existing results "
      "file was modified.")
    A("")
    A("## 1. Run metadata")
    A("")
    A(f"- Model: `{a['model']}`, reasoning effort `{a['effort']}`; snapshots "
      f"seen: {', '.join('`' + s + '`' for s in a['model_snapshots'])}")
    span = a.get("collection_span")
    A(f"- Collection span: {span[0]} → {span[1]}" if span else "- Collection span: n/a")
    A(f"- Trials: {a['trials']} per query per condition, trial-aligned; no "
      "sampling seeds (OpenAI sampling is not seedable; matches the top-20 runs).")
    A(f"- Window: {a['window']} passages, numbered 1..{a['window']} in input "
      "order; model returns a permutation (strict json_schema `{ranking: [int]}`).")
    A(f"- Prompt: identical objects to the top-20 harness "
      f"(`retrieval/listwise_rerank.py`); combined SHA-256 `{a['prompt_sha256']}`.")
    A("- Input orderings (never shuffled/re-sorted):")
    for c in CONDITIONS:
        A(f"  - Condition {c}: {COND_ORDERING[c]}")
    nA = {d: conds['A'][d]['n'] for d in BRIGHT_SUBSETS}
    nB = {d: conds['B'][d]['n'] for d in BRIGHT_SUBSETS}
    A(f"- Query counts — Condition B: {sum(nB.values())} "
      f"({', '.join(f'{d} {nB[d]}' for d in BRIGHT_SUBSETS)}). Condition A: "
      f"{sum(nA.values())} ({', '.join(f'{d} {nA[d]}' for d in BRIGHT_SUBSETS)}) "
      "— queries without cached zerank-2 scores (the documented per-provider "
      "drops) cannot form a Zerank top-100 and are excluded from A.")
    A(f"- Prices used: ${a['prices_per_1m']['input']:.2f} / "
      f"${a['prices_per_1m']['output']:.2f} per 1M input/output tokens "
      "(Luna list, effective 2026-07-30).")
    A("")

    A("## 2. Pilot results (incl. amendment-1 severity analysis)")
    A("")
    if PILOT_REPORT.exists():
        rep = json.loads(PILOT_REPORT.read_text())
        it = rep["input_tokens"]; lt = rep["latency_s"]
        gate = rep.get("gate", {})
        A(f"Pilot ({rep['ts']}): {rep['n_ok']}/{rep['n_planned']} calls ok, "
          f"{len(rep['api_errors'])} API errors. Malformed "
          f"{rep['malformed_count']}/{rep['n_ok']} = {rep['malformed_rate']:.1%} "
          "(gate suspended by amendment 1; see severity below). "
          f"Input tokens mean {it['mean']:,.0f} — within the amended expected "
          f"band {PILOT_TOKEN_BAND[0]:,}–{PILOT_TOKEN_BAND[1]:,} "
          f"(original spec estimate {it['spec_estimate']:,} assumed the d=500 "
          "modeling cap; actual BRIGHT docs are ~110–180 tokens, so the "
          f"{it['deviation_from_spec']:+.0%} deviation is explained); range "
          f"{it['min']:,}–{it['max']:,}. Latency median {lt['median']:.1f}s "
          f"(min {lt['min']:.1f} / max {lt['max']:.1f}). Amended gate: "
          f"{'PASSED' if gate.get('passed') else 'FAILED'}.")
        A("")
        A("Full per-call pilot detail: `results/raw/listwise_top100/pilot_report.json`.")
    else:
        A("_No pilot report found (pilot_report.json missing)._")
    A("")
    if SEVERITY_REPORT.exists():
        sev = json.loads(SEVERITY_REPORT.read_text())
        dr = sev["decision_rule"]; fr = sev["first_repair_rank"]
        A(f"**Severity analysis** ({sev['ts']}, over the {sev['n_responses']} "
          "pilot responses, pre-repair): "
          f"head-intact@20 {sev['head_intact_20_count']}/{sev['n_responses']} "
          f"= {sev['head_intact_20_rate']:.0%}; head-intact@10 "
          f"{sev['head_intact_10_count']}/{sev['n_responses']} = "
          f"{sev['head_intact_10_rate']:.0%}. First repair-affected rank over "
          f"the {sev['malformed_count']} malformed responses: min {fr['min']}, "
          f"median {fr['median']} (values: {fr['values']}). Truncation "
          f"lengths: {sev['truncation_lengths'] or '—'}. **Decision rule "
          f"(threshold head-intact@20 ≥ {dr['threshold_head_intact_20']:.0%}): "
          f"{dr['decision']}.**")
        A("")
        A("| Response | valid | n valid unique | head@20 | head@10 | 1st repair rank | raw len | flags |")
        A("|---|---|---|---|---|---|---|---|")
        for r in sev["per_response"]:
            A(f"| {r['domain']}/{r['condition']}/q{r['qid']} | "
              f"{'✓' if r['valid'] else '✗'} | {r['n_valid_unique']} | "
              f"{'✓' if r['head_intact_20'] else '✗'} | "
              f"{'✓' if r['head_intact_10'] else '✗'} | "
              f"{r['first_repair_rank'] if r['first_repair_rank'] is not None else '—'} | "
              f"{r['raw_len']} | {', '.join(r['flags']) or '—'} |")
        A("")

    A("## 3. Malformed-output statistics")
    A("")
    A("A response is **valid** iff it is an exact permutation of 1..100. "
      "Repair policy (applied mechanically): keep the first occurrence of each "
      "in-range ID, drop duplicates/out-of-range, append missing IDs at the "
      "end in input order. A malformed response can carry several flags.")
    A("")
    A("| Condition | Subset | Calls | Malformed (t0/t1/t2) | Rate | Flags | "
      "IDs appended | Head@20 | Head@10 | Med. 1st-repair rank |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for c in CONDITIONS:
        tot_calls = tot_mal = tot_app = tot_h20 = tot_h10 = 0
        for d in BRIGHT_SUBSETS:
            s = conds[c][d]
            sv = s["severity"]
            tot_calls += s["calls"]; tot_mal += s["malformed_total"]
            tot_app += s["appended_ids_total"]
            tot_h20 += round(sv["head_intact_20_rate"] * s["calls"])
            tot_h10 += round(sv["head_intact_10_rate"] * s["calls"])
            fl = ", ".join(f"{k}:{v}" for k, v in sorted(s["flag_counts"].items())) or "—"
            mbt = "/".join(str(x) for x in s["malformed_by_trial"])
            fr = sv["first_repair_rank_median"]
            A(f"| {c} | {d} | {s['calls']} | {s['malformed_total']} ({mbt}) | "
              f"{s['malformed_total']/s['calls']:.1%} | {fl} | "
              f"{s['appended_ids_total']} | {sv['head_intact_20_rate']:.1%} | "
              f"{sv['head_intact_10_rate']:.1%} | "
              f"{fr if fr is not None else '—'} |")
        A(f"| **{c}** | **all** | **{tot_calls}** | **{tot_mal}** | "
          f"**{tot_mal/tot_calls:.1%}** |  | **{tot_app}** | "
          f"**{tot_h20/tot_calls:.1%}** | **{tot_h10/tot_calls:.1%}** |  |")
    A("")
    A("Head-intact@K = the response contained ≥ K valid unique in-range IDs "
      "before any repair (amendment 1). Truncation lengths per condition are "
      "in `analysis.json` under `severity.truncation_lengths`.")
    A("")
    A("### Full-run severity per condition × trial (amendment-1 resolution)")
    A("")
    A("First-repair-affected-rank distribution: min / median / max over the "
      "malformed responses, plus counts by bucket (≤20 = repair touches the "
      "head the metrics score; 21–50; 51–98; ≥99 = tail-only).")
    A("")
    A("| Condition | Trial | Calls | Malformed | Head-intact@20 | "
      "1st-repair min/med/max | ≤20 | 21–50 | 51–98 | ≥99 |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for c in CONDITIONS:
        ct = fs.get("per_condition_trial", {}).get(c, {})
        for tr in sorted(ct, key=int):
            s = ct[tr]
            fr = s["first_repair_rank"]
            b = fr["buckets"]
            A(f"| {c} | {tr} | {s['calls']} | {s['malformed']} "
              f"({s['malformed_rate']:.1%}) | {s['head_intact_20']} "
              f"({s['head_intact_20_rate']:.1%}) | "
              f"{fr['min']}/{fr['median']}/{fr['max']} | "
              f"{b['le20']} | {b['21-50']} | {b['51-98']} | {b['ge99']} |")
    if oh is not None:
        all_calls = sum(fs["per_condition_trial"][c][t]["calls"]
                        for c in CONDITIONS for t in fs["per_condition_trial"][c])
        A(f"| **all** |  | **{all_calls}** |  | **{oh:.1%}** |  |  |  |  |  |")
    A("")
    A("Per-response head-intact@20 flags (every call): "
      "`results/raw/listwise_top100/severity_per_response.jsonl`.")
    A("")

    for c, title in (("A", "Condition A — Luna over the Zerank top-100"),
                     ("B", "Condition B — Luna over the hybrid top-100")):
        A(f"## {4 if c == 'A' else 5}. {title}")
        A("")
        A("Input-order row = the unreordered input list (the pipeline "
          "baseline). Luna rows are means across the 3 trials ± across-trial "
          "std. R@100 is the input pool recall by construction (verification "
          "below).")
        A("")
        hdr = "| Subset | n | " + " | ".join(
            f"{MET_LABEL[m]} in / Luna" for m in METRICS) + " |"
        A(hdr)
        A("|---|---|" + "---|" * len(METRICS))
        for d in BRIGHT_SUBSETS:
            s = conds[c][d]
            cells = [f"{fmt(s['baseline'][m])} / {fmt(s['mean'][m])} ± "
                     f"{s['std'][m]:.3f}" for m in METRICS]
            A(f"| {d} | {s['n']} | " + " | ".join(cells) + " |")
        mcells = [f"{fmt(meds[c]['baseline'][m])} / {fmt(meds[c]['mean'][m])} ± "
                  f"{meds[c]['std'][m]:.3f}" for m in METRICS]
        A(f"| **median** |  | " + " | ".join(f"**{x}**" for x in mcells) + " |")
        A("")
        worst = max(conds[c][d]["r100_verification_max_abs_diff"]
                    for d in BRIGHT_SUBSETS)
        A(f"R@100 verification: max |Luna R@100 − input R@100| across all "
          f"subsets × trials = {worst:.2e} (must be ~0; the ranking is a "
          "permutation of the input pool). Input-pool R@100 cross-subset "
          f"median = {meds[c]['baseline']['recall_at_100']:.3f}, matching the "
          f"spec's pinned {PINNED_INPUT_R100[c]:.3f} (checked before writing).")
        A("")
        A("Reference rows (cross-subset medians from the existing cached "
          "results; not recomputed):")
        A("")
        A("| Reference | nDCG@10 | S@1 | R@20 |")
        A("|---|---|---|---|")
        for label, vals in REFERENCE_ROWS:
            A(f"| {label} | "
              + " | ".join(fmt(vals[m]) if m in vals else "—"
                           for m in ("nDCG_at_10", "recall_at_1", "recall_at_20"))
              + " |")
        A("")

    A("## 6. Promotion / demotion decomposition")
    A("")
    A("Per query: **promoted** = gold docs with input rank 21–100 and output "
      "rank 1–20; **demoted** = gold docs with input rank 1–20 and output rank "
      "21–100. Counts are summed over queries (mean ± std across trials). "
      "Implied net ΔR@20 = mean over queries of (promoted−demoted)/|gold|; it "
      "equals the measured Luna−input R@20 delta by construction (cross-check "
      "column).")
    A("")
    for c in CONDITIONS:
        A(f"**Condition {c}:**")
        A("")
        A("| Subset | Promoted | Demoted | Implied net ΔR@20 | Measured ΔR@20 |")
        A("|---|---|---|---|---|")
        tp, td, weighted = 0.0, 0.0, 0.0
        n_all = 0
        for d in BRIGHT_SUBSETS:
            p = conds[c][d]["promotion"]; n = conds[c][d]["n"]
            A(f"| {d} | {p['promoted_mean']:.1f} ± {p['promoted_std']:.1f} | "
              f"{p['demoted_mean']:.1f} ± {p['demoted_std']:.1f} | "
              f"{p['net_r20_mean']:+.3f} ± {p['net_r20_std']:.3f} | "
              f"{p['measured_r20_delta']:+.3f} |")
            tp += p["promoted_mean"]; td += p["demoted_mean"]
            weighted += p["net_r20_mean"] * n; n_all += n
        A(f"| **overall (pooled)** | **{tp:.1f}** | **{td:.1f}** | "
          f"**{weighted/n_all:+.3f}** |  |")
        A("")

    A("## 7. Token counts and actual cost")
    A("")
    A("| Condition | Calls | Input tok | Output tok | Reasoning tok | Cost |")
    A("|---|---|---|---|---|---|")
    gi = go = gc = 0
    gu = 0.0
    for c in CONDITIONS:
        ti = sum(conds[c][d]["tokens"]["input"] for d in BRIGHT_SUBSETS)
        to = sum(conds[c][d]["tokens"]["output"] for d in BRIGHT_SUBSETS)
        tr_ = sum(conds[c][d]["tokens"]["reasoning"] for d in BRIGHT_SUBSETS)
        nc = sum(conds[c][d]["calls"] for d in BRIGHT_SUBSETS)
        usd = (ti * PRICE_IN + to * PRICE_OUT) / 1e6
        A(f"| {c} | {nc} | {ti:,} | {to:,} | {tr_:,} | ${usd:.2f} |")
        gi += ti; go += to; gc += nc; gu += usd
    A(f"| **total** | **{gc}** | **{gi:,}** | **{go:,}** |  | **${gu:.2f}** |")
    A("")
    A(f"Per-query cost (both conditions, 3 trials): ~${gu / (sum(nB.values()) + sum(nA.values())):.4f}. "
      f"Prices: ${PRICE_IN:.2f}/${PRICE_OUT:.2f} per 1M in/out (Luna list, "
      "2026-07-30). Reasoning tokens are included in output tokens (effort "
      "`none` ⇒ expected ≈0).")
    A("")

    A("## 8. Raw per-query results")
    A("")
    A("- Response caches (append-only JSONL, one line per (query, trial), with "
      "raw + repaired rankings, flags, tokens, latency, model snapshot): "
      "`results/raw/listwise_top100/cache/{subset}__{zerank_top100|hybrid_top100}"
      f"__{MODEL}__{EFFORT}.jsonl`")
    A("- Input pools (per-query 100-doc lists in input order + gold + texts): "
      "`results/raw/listwise_top100/pools/`")
    A("- Per-response severity flags (valid / head-intact@20 / first-repair "
      "rank per (condition, subset, query, trial)): "
      "`results/raw/listwise_top100/severity_per_response.jsonl`")
    A("- Machine-readable analysis: `results/raw/listwise_top100/analysis.json`; "
      "pilot: `results/raw/listwise_top100/pilot_report.json`; pilot severity: "
      "`results/raw/listwise_top100/severity_report.json`.")
    A("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-pools", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--severity", action="store_true",
                    help="Amendment-1 pilot severity analysis (zero API calls).")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="Skip the pilot-gate check before --run.")
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    args = ap.parse_args()
    if args.build_pools:
        build_pools()
    elif args.dry_run:
        dry_run()
    elif args.pilot:
        run_pilot(args.concurrency)
    elif args.severity:
        run_severity()
    elif args.run:
        run_full(args.concurrency, args.force)
    elif args.analyze:
        analyze()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
