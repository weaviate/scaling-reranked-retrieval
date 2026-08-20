"""The experiment's condition menu — equal-weight fusion only.

Menu policy (2026-07-10): the experiments test EQUAL-WEIGHT fusion only. The
menu is the no-rerank baseline, the three singletons, the three equal-weight
pairs (x rrf/rsf), and the equal-weight 3-way (x rrf/rsf) — 12 conditions.
Weighted tilts (0.7/0.3 pairs, 0.5/0.25/0.25 3-ways) were removed from the
project entirely after the fixed-weight analysis found no tilt beats equal
beyond noise — no code generates or sweeps tilted conditions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

# Mirrors src.adapters.retrieval.providers.{Provider,FusionMethod}. Defined
# locally (types only) because importing anything under
# src/adapters/retrieval/ triggers its __init__ re-exports, which pull the
# weaviate client into every consumer — and the domain layer must stay
# provider-import-free (only src.application.collect may import clients).
Provider = Literal["cohere", "voyage", "zerank", "hybrid"]
FusionMethod = Literal["rrf", "rsf"]


@dataclass
class Condition:
    name: str
    provider: Optional[Provider] = None  # None => no reranking, plain hybrid retrieval
    fusion_method: Optional[FusionMethod] = None
    weights: Optional[dict] = None
    # Which rerankers participate when provider == "hybrid".
    rerankers: tuple[str, ...] = ("cohere", "voyage")


@dataclass
class _Condition:
    """Minimal condition stand-in for ad-hoc derivations.

    Only the attributes DerivedSearchAgent reads are present (no name). Using
    the real agent (rather than re-implementing the sort) guarantees the pool
    filtering and tie-breaking match the derive path exactly.
    """

    provider: Optional[str]
    fusion_method: Optional[str] = None
    weights: Optional[dict] = None
    rerankers: tuple[str, ...] = ()


SINGLETON_CONDITIONS = {"cohere_only", "voyage_only", "zerank_only"}
BASELINE_CONDITION = "hybrid_only"

# Pair families: (name tag, (reranker a, reranker b)), alphabetical within pair.
_PAIRS = (
    ("cv", ("cohere", "voyage")),
    ("cz", ("cohere", "zerank")),
    ("vz", ("voyage", "zerank")),
)
_THREE_WAY = ("cohere", "voyage", "zerank")


def is_equal_weight(condition: Condition) -> bool:
    """True for non-fusion conditions and fusions with uniform weights."""
    if condition.weights is None:
        return True
    return len(set(condition.weights.values())) == 1


def build_menu() -> list[Condition]:
    """Generate the 12-condition equal-weight menu.

    Ordering: baselines/singletons, then per pair family (equal rrf, equal
    rsf), then the equal 3-way (rrf, rsf). Condition names and relative order
    match every runs file under results/ — do not reorder.
    """
    menu = [
        Condition("hybrid_only", provider=None),
        Condition("cohere_only", provider="cohere"),
        Condition("voyage_only", provider="voyage"),
        Condition("zerank_only", provider="zerank"),
    ]
    for tag, (a, b) in _PAIRS:
        for method in ("rrf", "rsf"):
            menu.append(
                Condition(
                    f"{method}_{tag}_equal",
                    provider="hybrid",
                    fusion_method=method,
                    weights={a: 0.5, b: 0.5},
                    rerankers=(a, b),
                )
            )
    for method in ("rrf", "rsf"):
        menu.append(
            Condition(
                f"{method}_equal_3way",
                provider="hybrid",
                fusion_method=method,
                weights={r: 1 / 3 for r in _THREE_WAY},
                rerankers=_THREE_WAY,
            )
        )
    return menu


CONDITIONS: list[Condition] = build_menu()
