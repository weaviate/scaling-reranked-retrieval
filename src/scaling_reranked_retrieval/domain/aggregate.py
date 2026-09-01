"""Aggregation conventions: MEAN across queries (per-query recall is too
coarse for a median), then MEDIAN across subsets."""
from __future__ import annotations

import statistics
from typing import Iterable


def qmean(xs: list[float]) -> float:
    """Mean across queries."""
    return statistics.fmean(xs) if xs else 0.0



def median_across_subsets(values: Iterable[float]) -> float:
    """Median across per-subset aggregates (the cross-domain rollup)."""
    return statistics.median(values)
