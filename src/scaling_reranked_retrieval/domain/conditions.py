"""The experiment's 12-condition menu — equal-weight fusion only."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

# Redefined locally (mirrors adapters.retrieval.providers) so the domain layer stays free of adapter imports.
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
    """Minimal condition stand-in for ad-hoc derivations (only the attributes DerivedSearchAgent reads)."""

    provider: Optional[str]
    fusion_method: Optional[str] = None
    weights: Optional[dict] = None
    rerankers: tuple[str, ...] = ()


SINGLETON_CONDITIONS = {"cohere_only", "voyage_only", "zerank_only"}
BASELINE_CONDITION = "hybrid_only"

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

    Names and relative order match every runs file under results/ — do not reorder.
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
