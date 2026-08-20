#!/usr/bin/env python3
"""Reranker agreement / decorrelation analysis over the k=2000 caches.

Run script — a thin wrapper; all logic lives in src.application.analysis.agreement.
Usage: uv run python scripts/agreement.py [args]  (-h for options)
"""
from src.application.analysis.agreement import main

if __name__ == "__main__":
    main()
