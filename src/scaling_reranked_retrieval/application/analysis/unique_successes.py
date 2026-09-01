"""Per-reranker unique successes at k=200.

Unique success @K = a reranker lands a gold doc in its top-K while both others
miss (Success@K = recall@K > 0). Computed over the all-three-present
intersection — required for "unique" to be well-defined; zero reranker API calls.
"""
from __future__ import annotations

import argparse
import json

from scaling_reranked_retrieval.application.derived import DerivedSearchAgent
from scaling_reranked_retrieval.domain.conditions import CONDITIONS as _RE_CONDITIONS
from scaling_reranked_retrieval.config import MODEL_OVERRIDES, PROVIDERS, RESULTS_DIR
from scaling_reranked_retrieval.application.queryset import load_and_validate

from scaling_reranked_retrieval.adapters import qab

qab.setup()

RETRIEVED_K = 200
RERANKED_K = 20                  # deployed output cap; success cutoffs are <= 20.
CACHE_K = 2000
SUCCESS_CUTOFFS = (1, 5, 20)
SUBSETS = ["biology", "earth_science", "economics", "psychology", "robotics"]

# Taken straight from CONDITIONS so rankings are byte-identical to the harness.
_SINGLETON_BY_PROVIDER = {
    c.provider: c
    for c in _RE_CONDITIONS
    if c.name in ("cohere_only", "voyage_only", "zerank_only")
}


def _singleton_top20(cache, query: str, provider: str, k: int, reranked_k: int) -> list[str]:
    agent = DerivedSearchAgent(
        cache=cache,
        retrieved_k=k,
        condition=_SINGLETON_BY_PROVIDER[provider],
        reranked_k=reranked_k,
    )
    return [o.object_id for o in agent.run(query)]


def analyze_subset(cache, qs, k: int, reranked_k: int) -> dict:
    """Per-(cutoff, provider) success / unique-success counts + query lists."""
    queries = list(qs.gold.keys())
    n = len(queries)

    counts = {
        K: {p: {"success": 0, "unique": 0, "unique_queries": []} for p in PROVIDERS}
        for K in SUCCESS_CUTOFFS
    }
    coverage = {K: {"any_hit": 0, "all_miss": 0} for K in SUCCESS_CUTOFFS}

    for text in queries:
        gold = qs.gold[text]
        qid = qs.query_ids.get(text, text[:64])
        top20 = {p: _singleton_top20(cache, text, p, k, reranked_k) for p in PROVIDERS}
        for K in SUCCESS_CUTOFFS:
            hit = {p: bool(set(top20[p][:K]) & gold) for p in PROVIDERS}
            n_hits = sum(hit.values())
            if n_hits:
                coverage[K]["any_hit"] += 1
            else:
                coverage[K]["all_miss"] += 1
            for p in PROVIDERS:
                if hit[p]:
                    counts[K][p]["success"] += 1
                    if n_hits == 1:
                        counts[K][p]["unique"] += 1
                        counts[K][p]["unique_queries"].append(
                            {"query_id": qid, "query": text}
                        )

    return {"n_queries_intersection": n, "counts": counts, "coverage": coverage}


def run(k: int, reranked_k: int, list_cutoff: int, write: bool = True) -> dict:
    per_subset: dict[str, dict] = {}
    for ds in SUBSETS:
        print(f"\n=== {ds} (k={k}) ===")
        loaded = load_and_validate(ds)
        if loaded is None:
            print(f"  [skip] no usable cache for {ds}")
            continue
        cache, qs = loaded
        res = analyze_subset(cache, qs, k, reranked_k)
        per_subset[ds] = res
        u = {p: res["counts"][20][p]["unique"] for p in PROVIDERS}
        print(f"  n={res['n_queries_intersection']}  unique@20 {u}")

    present = list(per_subset.keys())
    if not present:
        raise SystemExit("No subsets had a usable k=2000 cache.")

    # Aggregate sums absolute query tallies across subsets, not means.
    aggregate = {
        K: {
            **{
                p: {
                    "success": sum(per_subset[d]["counts"][K][p]["success"] for d in present),
                    "unique": sum(per_subset[d]["counts"][K][p]["unique"] for d in present),
                }
                for p in PROVIDERS
            },
            "any_hit": sum(per_subset[d]["coverage"][K]["any_hit"] for d in present),
            "all_miss": sum(per_subset[d]["coverage"][K]["all_miss"] for d in present),
            "n": sum(per_subset[d]["n_queries_intersection"] for d in present),
        }
        for K in SUCCESS_CUTOFFS
    }

    payload = {
        "k": k,
        "reranked_k": reranked_k,
        "cache_retrieved_k": CACHE_K,
        "success_cutoffs": list(SUCCESS_CUTOFFS),
        "model_overrides": MODEL_OVERRIDES,
        "definition": (
            "unique success @K = reranker lands a gold doc in its top-K while "
            "both other rerankers miss; computed over the all-three-present "
            "intersection at retrieved_k={k}, reranked_k={rk}."
        ).format(k=k, rk=reranked_k),
        "per_subset": per_subset,
        "aggregate": aggregate,
    }

    if write:
        out_dir = RESULTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"unique_successes_k{k}.json"
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote {json_path}")
        md_path = out_dir / f"unique_successes_k{k}_table.md"
        md_path.write_text(render_table(payload, present, list_cutoff))
        print(f"Wrote {md_path}")

    return payload


def render_table(payload: dict, present: list[str], list_cutoff: int) -> str:
    k = payload["k"]
    lines: list[str] = []
    A = lines.append

    A(f"# Per-Reranker Unique Successes at k={k}")
    A("")
    A(
        f"A reranker has a **unique success @K** on a query when it lands a gold "
        f"doc in its top-K while both other rerankers miss. Derived from the k="
        f"{payload['cache_retrieved_k']} caches at retrieved_k={k}, reranked_k="
        f"{payload['reranked_k']}, over the all-three-present intersection, zero "
        "reranker API calls. Counts are absolute query tallies; `(frac)` is the "
        "fraction of that subset's intersection queries."
    )
    A("")

    for K in payload["success_cutoffs"]:
        A(f"## Unique successes @{K}")
        A("")
        A("| Subset | n | "
          + " | ".join(f"{p} unique" for p in PROVIDERS)
          + " | ≥1 hit | all miss |")
        A("|---|---|" + "|".join(["---"] * (len(PROVIDERS) + 2)) + "|")
        for ds in present:
            res = payload["per_subset"][ds]
            n = res["n_queries_intersection"]
            cells = []
            for p in PROVIDERS:
                u = res["counts"][K][p]["unique"]
                cells.append(f"{u} ({u / n:.3f})")
            cov = res["coverage"][K]
            A(f"| {ds} | {n} | " + " | ".join(cells)
              + f" | {cov['any_hit']} | {cov['all_miss']} |")
        agg = payload["aggregate"][K]
        nA = agg["n"]
        acells = [f"**{agg[p]['unique']} ({agg[p]['unique'] / nA:.3f})**" for p in PROVIDERS]
        A(f"| **aggregate** | **{nA}** | " + " | ".join(acells)
          + f" | **{agg['any_hit']}** | **{agg['all_miss']}** |")
        A("")

    A(f"## Unique-success queries @{list_cutoff}")
    A("")
    A(f"The actual queries each reranker uniquely rescues at K={list_cutoff} "
      "(the other two miss). Query text truncated to 100 chars.")
    A("")
    for ds in present:
        res = payload["per_subset"][ds]
        bucket = res["counts"][list_cutoff]
        A(f"### {ds}")
        A("")
        for p in PROVIDERS:
            qs_list = bucket[p]["unique_queries"]
            A(f"**{p}** ({len(qs_list)}):")
            if not qs_list:
                A("- _none_")
            for q in qs_list:
                snippet = q["query"].replace("\n", " ").strip()[:100]
                A(f"- `{q['query_id']}` — {snippet}")
            A("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=RETRIEVED_K,
                        help="retrieved_k operating point (default 200).")
    parser.add_argument("--reranked-k", type=int, default=RERANKED_K,
                        help="Output cap for the rankings (default 20).")
    parser.add_argument("--list-cutoff", type=int, default=20,
                        choices=SUCCESS_CUTOFFS,
                        help="Which cutoff's unique-success queries to list in the md (default 20).")
    args = parser.parse_args()
    run(args.k, args.reranked_k, args.list_cutoff)


if __name__ == "__main__":
    main()
