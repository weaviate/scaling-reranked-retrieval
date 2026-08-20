"""Loader/validator for listwise-LLM ranking caches.

The listwise experiment (retrieval/listwise_rerank.py) caches every
(query, trial) LLM response — including the full reordered ranking
(`listwise_doc_ids`) — to an append-only JSONL keyed by
(domain, model, effort, pool-slug, first_k, pool_k):

    results/listwise/cache/<domain>__<model>__<effort>__<pool-slug>__first<K>__top<P>.jsonl

Those JSONLs are the canonical ranking store (analogous to caches/k{N}.json
for the cross-encoders). This module loads one into a validated
ListwiseRankings — completeness (every pool query × every trial), permutation
integrity (each ranking is exactly the pool's doc set), parse-failure flags —
so the fusion analysis (analysis/listwise_fusion.py) never touches raw JSONL
or filename conventions directly. Zero network.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.config import RESULTS_DIR

LISTWISE_DIR = RESULTS_DIR / "listwise"

# Short condition-name labels for known listwise models, shared by every
# listwise analysis (fusion / unique-success / oracle-routing) so condition
# names and output filenames stay consistent across reports. Unknown models
# fall back to a sanitized model id.
SHORT_LABEL = {
    "gpt-5.4-mini": "mini",
    "gpt-5.4": "full",
    "gpt-5.6-luna": "luna",
    "gpt-5.6-terra": "terra",
}


def short_label(model: str) -> str:
    return SHORT_LABEL.get(model, model.replace("/", "-"))


def pool_key(pool_source: str, first_k: int, pool_k: int) -> str:
    """Canonical pool-config slug used in filenames and run dirs."""
    return f"{pool_source.replace('_', '-')}__first{first_k}__top{pool_k}"


def pool_file(domain: str, pool_source: str, first_k: int, pool_k: int) -> Path:
    return LISTWISE_DIR / "pools" / f"{domain}__{pool_key(pool_source, first_k, pool_k)}.json"


def ranking_cache_file(
    domain: str, model: str, effort: str, pool_source: str, first_k: int, pool_k: int
) -> Path:
    return (
        LISTWISE_DIR / "cache" /
        f"{domain}__{model}__{effort}__{pool_key(pool_source, first_k, pool_k)}.jsonl"
    )


@dataclass
class ListwiseRankings:
    """Validated per-(domain, model) listwise rankings over a fixed pool."""

    domain: str
    model: str
    effort: str
    pool_source: str
    first_k: int
    pool_k: int
    trials: int
    pool_meta: dict = field(default_factory=dict)
    # All maps are keyed by qid.
    gold: dict[str, list[str]] = field(default_factory=dict)
    pool_order: dict[str, list[str]] = field(default_factory=dict)  # pool-source baseline ranking
    rankings: dict[str, list[list[str]]] = field(default_factory=dict)  # qid -> [per-trial ranking]
    parse_failed: dict[str, list[bool]] = field(default_factory=dict)

    @property
    def qids(self) -> list[str]:
        return sorted(self.pool_order.keys())


def load_pool(domain: str, pool_source: str, first_k: int, pool_k: int) -> dict:
    path = pool_file(domain, pool_source, first_k, pool_k)
    if not path.exists():
        raise FileNotFoundError(
            f"No pool file at {path}. Build it with: uv run python "
            f"retrieval/listwise_rerank.py --build-all-pools "
            f"--pool-source {pool_source} --first-stage-k {first_k} --pool-k {pool_k}"
        )
    with open(path) as f:
        return json.load(f)


def load_listwise_rankings(
    domain: str,
    model: str,
    effort: str = "medium",
    pool_source: str = "zerank_only",
    first_k: int = 2000,
    pool_k: int = 20,
    trials: int = 3,
) -> ListwiseRankings:
    """Load + validate one (domain, model) ranking cache against its pool.

    Raises with a precise message if the cache is missing, incomplete (any
    (query, trial) not cached), or corrupt (a ranking that is not a
    permutation of its query's pool). Duplicate (qid, trial) JSONL lines are
    resolved last-wins, matching the experiment's own cache loader.
    """
    pool = load_pool(domain, pool_source, first_k, pool_k)
    cache_path = ranking_cache_file(domain, model, effort, pool_source, first_k, pool_k)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"No ranking cache at {cache_path}. Collect it with: uv run python "
            f"retrieval/listwise_rerank.py --all-domains --model {model} "
            f"--reasoning-effort {effort} --pool-source {pool_source} "
            f"--first-stage-k {first_k} --pool-k {pool_k} --trials {trials}"
        )

    out = ListwiseRankings(
        domain=domain, model=model, effort=effort, pool_source=pool_source,
        first_k=first_k, pool_k=pool_k, trials=trials,
        pool_meta=pool.get("metadata", {}),
    )
    for q in pool["queries"].values():
        qid = q["qid"]
        out.gold[qid] = list(q["gold"])
        out.pool_order[qid] = list(q["doc_ids"])

    # Last-wins per (qid, trial), matching load_query_cache in the experiment.
    by_key: dict[tuple[str, int], dict] = {}
    with open(cache_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            by_key[(str(entry["qid"]), int(entry["trial"]))] = entry

    missing: list[tuple[str, int]] = []
    for qid in out.pool_order:
        per_trial: list[list[str]] = []
        failed: list[bool] = []
        pool_set = set(out.pool_order[qid])
        for t in range(trials):
            entry = by_key.get((qid, t))
            if entry is None:
                missing.append((qid, t))
                continue
            ranking = list(entry["listwise_doc_ids"])
            if set(ranking) != pool_set or len(ranking) != len(pool_set):
                raise ValueError(
                    f"{cache_path.name}: ranking for (qid={qid}, trial={t}) is "
                    f"not a permutation of the pool ({len(ranking)} docs vs "
                    f"pool {len(pool_set)})."
                )
            per_trial.append(ranking)
            failed.append(bool(entry.get("parse_failed", False)))
        out.rankings[qid] = per_trial
        out.parse_failed[qid] = failed

    if missing:
        raise ValueError(
            f"{cache_path.name}: incomplete — {len(missing)} missing "
            f"(query, trial) cells (first few: {missing[:5]}). Re-run the "
            "listwise experiment to fill them (resumable, only missing calls "
            "are paid)."
        )
    return out
