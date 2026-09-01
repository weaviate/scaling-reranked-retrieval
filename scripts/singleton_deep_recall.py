#!/usr/bin/env python3
"""Singleton R@K grid over one denominator.

Run script — a thin wrapper; all logic lives in scaling_reranked_retrieval.application.analysis.singleton_deep_recall.
Usage: uv run python scripts/singleton_deep_recall.py [args]  (-h for options)
"""
from scaling_reranked_retrieval.application.analysis.singleton_deep_recall import main

if __name__ == "__main__":
    main()
