"""Aggregation conventions, implemented once.

The experiment's standard rollup is: MEAN across queries (per-query recall@K
is too coarse for a median — mostly 0, else 1/|gold|, so a query-median
collapses to 0/1), then MEDIAN across subsets, then (noise-null only)
mean + band across seeds.
"""
from __future__ import annotations

import statistics
from typing import Iterable


def qmean(xs: list[float]) -> float:
    """Mean across queries. See module docstring for why not median."""
    return statistics.fmean(xs) if xs else 0.0



def median_across_subsets(values: Iterable[float]) -> float:
    """Median across per-subset aggregates (the cross-domain rollup)."""
    return statistics.median(values)
