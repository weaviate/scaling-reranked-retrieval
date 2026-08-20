#!/usr/bin/env python3
"""Per-model unique successes at the listwise tier.

Run script — a thin wrapper; all logic lives in src.application.analysis.listwise_unique_successes.
Usage: uv run python scripts/listwise_unique_successes.py [args]  (-h for options)
"""
from src.application.analysis.listwise_unique_successes import main

if __name__ == "__main__":
    main()
