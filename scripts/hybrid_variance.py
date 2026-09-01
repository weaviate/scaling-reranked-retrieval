#!/usr/bin/env python3
"""Across-trial variance of first-stage hybrid retrieval (LIVE Weaviate calls).

Run script — a thin wrapper; all logic lives in scaling_reranked_retrieval.application.experiments.hybrid_variance.
Usage: uv run python scripts/hybrid_variance.py [args]  (-h for options)
"""
from scaling_reranked_retrieval.application.experiments.hybrid_variance import main

if __name__ == "__main__":
    main()
