#!/usr/bin/env python3
"""Per-reranker unique-success analysis.

Run script — a thin wrapper; all logic lives in src.application.analysis.unique_successes.
Usage: uv run python scripts/unique_successes.py [args]  (-h for options)
"""
from src.application.analysis.unique_successes import main

if __name__ == "__main__":
    main()
