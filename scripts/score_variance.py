#!/usr/bin/env python3
"""Reranker determinism check (LIVE reranker calls).

Run script — a thin wrapper; all logic lives in src.application.experiments.score_variance.
Usage: uv run python scripts/score_variance.py [args]  (-h for options)
"""
from src.application.experiments.score_variance import main

if __name__ == "__main__":
    main()
