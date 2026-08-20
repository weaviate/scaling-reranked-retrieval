"""Success@20 for selected conditions, derived from the k=2000 score caches.

Success@20 (hit rate) = fraction of queries with AT LEAST ONE gold doc in the
top-20 of a condition's reranked output. This differs from the published
Recall@20, which (for multi-gold BRIGHT queries) is |gold ∩ top-20| / |gold|
per query — a mean recall of 0.65 does NOT tell you what fraction of queries
got at least one gold into the pool. qab computes both; success was simply
never in the `extra_metrics` list for these runs, and the runs/*.json files
only persist aggregates, so it must be re-derived from the caches.

Pure derivation — ZERO reranker API calls. Reuses:
  - DerivedSearchAgent (byte-identical to the derive path behind runs/*.json)
  - qab's calculate_success_at_k / calculate_recall_at_k (exact-match metrics)
  - the same coverage denominator as run_search_eval: queries scored by EVERY
    provider the condition uses (a dropped query has no `{p}_scores` key in
    the cache, so the derive path errors and qab skips it — singletons cover
    their own present set, fusion conditions the participating intersection).

Reproducibility: RSF conditions break exact fused-score ties in `set(pool)`
iteration order, which is `PYTHONHASHSEED`-dependent (~1 query of wobble; see
CLAUDE.md). This script re-execs with PYTHONHASHSEED=0 so its own outputs are
bit-reproducible; against the published runs (written under random seeds) the
regression guard therefore tolerates ~1 query on rsf_* conditions and demands
exact equality everywhere else.

Outputs:
  results/success_at_20.json
  results/success_at_20_table.md

Usage:
    uv run python scripts/success_at_20.py
    uv run python scripts/success_at_20.py --conditions zerank_only
    uv run python scripts/success_at_20.py --datasets biology
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

# RSF tie-breaking is hash-seed-dependent (set iteration); pin for
# bit-reproducible output. Must happen before heavy imports.
if os.environ.get("PYTHONHASHSEED") != "0":
    env = dict(os.environ, PYTHONHASHSEED="0")
    os.execve(sys.executable, [sys.executable, *sys.argv], env)

from query_agent_benchmarking.internal.adapters.dataset import (
    in_memory_dataset_loader,
)
from query_agent_benchmarking.internal.adapters.metrics.ir_metrics import (
    calculate_recall_at_k,
    calculate_success_at_k,
)

from src.application.derived import DerivedSearchAgent  # noqa: E402
from src.adapters.cache import ScoreCache, validate_cache_for_use  # noqa: E402
from src.domain.conditions import CONDITIONS  # noqa: E402
from src.config import (  # noqa: E402
    DATASETS,
    MODEL_OVERRIDES,
    RESULTS_DIR,
    get_results_dir,
)

from src.adapters import qab  # noqa: E402

qab.setup()

K_VALUES = (100, 200, 500, 1000, 2000)
CACHE_K = 2000
RERANKED_K = 20
CUTOFF = 20

CONDITION_BY_NAME = {c.name: c for c in CONDITIONS}
DEFAULT_CONDITIONS = ("zerank_only", "rsf_equal_3way")

# BRIGHT subsets with a k=2000 cache on disk (irpapers_text pending).
DEFAULT_DATASETS = ("biology", "earth_science", "economics", "psychology", "robotics")


def providers_needed(cond) -> tuple[str, ...]:
    """Providers whose scores the derive path requires for this condition."""
    if cond.provider is None:
        return ()
    if cond.provider == "hybrid":
        return tuple(cond.rerankers)
    return (cond.provider,)


def published_recall_at_20(slug: str, k: int, cond_name: str):
    """Read a condition's avg_recall_at_20_mean from the on-disk run summaries.

    Returns (value, source_filename). Prefers runs/k{k}_from_k2000.json, then
    any runs/k{k}_from_k*.json (biology's published small-k sweeps derive from
    the older k500 cache — those can differ from a k2000-cache derivation by a
    slightly different hybrid pool snapshot), then runs_rk100/ (psychology and
    robotics were run directly at reranked_k=100; R@20 is byte-identical).
    """
    results_dir = get_results_dir(slug)
    candidates = [results_dir / "runs" / f"k{k}_from_k{CACHE_K}.json"]
    candidates += sorted((results_dir / "runs").glob(f"k{k}_from_k*.json"))
    candidates += sorted((results_dir / "runs_rk100").glob(f"k{k}_from_k*.json"))
    for path in candidates:
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        entry = data.get("results", {}).get(cond_name)
        if entry is not None:
            return float(entry["avg_recall_at_20_mean"]), path.name
    return None, None


def analyze_dataset(slug: str, cond_names: list[str]) -> dict:
    cfg = DATASETS[slug]
    cache_path = get_results_dir(slug) / "caches" / f"k{CACHE_K}.json"
    cache = ScoreCache.load(cache_path)
    validate_cache_for_use(
        cache,
        needed_retrieved_k=CACHE_K,
        expected_model_overrides=MODEL_OVERRIDES,
        expected_dataset=cfg.qab_name,
        expected_collection=cfg.collection,
    )

    _, queries = in_memory_dataset_loader(cfg.qab_name, queries_only=True)
    gold_by_text = {q.question: [str(d) for d in q.dataset_ids] for q in queries}

    out: dict[str, dict] = {}
    for cond_name in cond_names:
        cond = CONDITION_BY_NAME[cond_name]
        needed = providers_needed(cond)
        # Coverage: every needed provider scored the query (mirrors qab's
        # skip-empty-result rule — dropped queries have no {p}_scores key).
        covered = [
            text
            for text, entry in cache.queries.items()
            if text in gold_by_text
            and all(entry.get(f"{p}_scores") for p in needed)
        ]

        per_k: dict[str, dict] = {}
        for k in K_VALUES:
            agent = DerivedSearchAgent(
                cache=cache, retrieved_k=k, condition=cond, reranked_k=RERANKED_K
            )
            successes, recalls = [], []
            for text in covered:
                ranked = [o.object_id for o in agent.run(text)]
                gold = gold_by_text[text]
                successes.append(calculate_success_at_k(gold, ranked, CUTOFF))
                recalls.append(calculate_recall_at_k(gold, ranked, CUTOFF))
            n = len(covered)
            success_at_20 = sum(successes) / n
            recall_at_20 = sum(recalls) / n
            published, source = published_recall_at_20(slug, k, cond_name)
            same_cache = bool(source) and f"from_k{CACHE_K}" in (source or "")
            # rsf_* published values carry ~1 query of PYTHONHASHSEED
            # tie-break wobble (CLAUDE.md); everything else must match exactly.
            tol = (2.0 / n) if cond_name.startswith("rsf") else 1e-9
            delta = None if published is None else recall_at_20 - published
            match = delta is not None and abs(delta) <= tol
            per_k[str(k)] = {
                "n_covered": n,
                "success_at_20": success_at_20,
                "recall_at_20_recomputed": recall_at_20,
                "recall_at_20_published": published,
                "published_source": source,
                "published_same_cache": same_cache,
                "regression_match": match,
            }
            if match:
                flag = "OK" if delta == 0 else "OK~tie"
            else:
                flag = "MISMATCH" if same_cache else "SNAPSHOT-DIFF"
            print(
                f"{cond_name:16s} {slug:14s} k={k:<5d} n={n:3d}  "
                f"success@20={success_at_20:.4f}  recall@20={recall_at_20:.4f}  "
                f"published={published} ({source})  [{flag}]"
            )
        out[cond_name] = {"n_covered": len(covered), "per_k": per_k}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Success@20 for selected conditions, derived from the k=2000 caches."
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
    args = parser.parse_args()
    for name in args.conditions:
        if name not in CONDITION_BY_NAME:
            parser.error(f"Unknown condition {name!r}")

    results = {slug: analyze_dataset(slug, args.conditions) for slug in args.datasets}

    out = {
        "conditions": args.conditions,
        "reranked_k": RERANKED_K,
        "cutoff": CUTOFF,
        "cache_k": CACHE_K,
        "model_overrides": MODEL_OVERRIDES,
        "pythonhashseed": "0",
        "denominator": (
            "queries scored by every provider the condition uses "
            "(matches runs/*.json per-condition coverage)"
        ),
        "datasets": results,
    }

    results_root = RESULTS_DIR
    json_path = results_root / "success_at_20.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    lines = [
        "# Success@20 (hit rate: >=1 gold doc in the reranked top-20)",
        "",
        "Derived from `caches/k2000.json` per dataset (zero API calls,",
        "PYTHONHASHSEED=0), same `DerivedSearchAgent` path and coverage",
        "denominator as `runs/*.json`. Recall@20 recomputed alongside matches",
        "the published values exactly wherever the published run derives from",
        "the k2000 cache (regression guard) — except biology's published",
        "k=100/500, which derive from the older k500 cache (slightly different",
        "hybrid pool snapshot, deltas <=0.005), and `rsf_*` conditions, whose",
        "published values carry ~1 query of hash-seed tie-break wobble.",
    ]
    for cond_name in args.conditions:
        lines += [
            "",
            f"## `{cond_name}`",
            "",
            "| Dataset | n | " + " | ".join(f"k={k}" for k in K_VALUES) + " |",
            "|---|---|" + "---|" * len(K_VALUES),
        ]
        for slug in args.datasets:
            res = results[slug][cond_name]
            cells = []
            for k in K_VALUES:
                cell = res["per_k"][str(k)]
                guard = (
                    ""
                    if (cell["regression_match"] or not cell["published_same_cache"])
                    else " ⚠️"
                )
                cells.append(
                    f"{cell['success_at_20']:.3f} (R@20 {cell['recall_at_20_recomputed']:.3f}){guard}"
                )
            lines.append(f"| {slug} | {res['n_covered']} | " + " | ".join(cells) + " |")
        med = {
            k: statistics.median(
                results[slug][cond_name]["per_k"][str(k)]["success_at_20"]
                for slug in args.datasets
            )
            for k in K_VALUES
        }
        lines.append(
            "| **median across subsets** | — | "
            + " | ".join(f"**{med[k]:.3f}**" for k in K_VALUES)
            + " |"
        )
    lines += [
        "",
        "Notes:",
        "- Success@20 = mean over covered queries of 1[top-20 contains >=1 gold].",
        "- Denominators differ per condition (own coverage), matching runs/*.json —",
        "  cross-condition deltas on drop-affected subsets (robotics) shift ~1 query.",
        "- Published `recall_at_1` is already Success@1 (qab computes success at k=1).",
    ]
    md_path = results_root / "success_at_20_table.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {json_path}\nWrote {md_path}")


if __name__ == "__main__":
    main()
