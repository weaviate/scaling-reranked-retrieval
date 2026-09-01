#!/usr/bin/env python3
"""Success@20 extraction.

Run script — a thin wrapper; all logic lives in scaling_reranked_retrieval.application.analysis.success_at_20.
Usage: uv run python scripts/success_at_20.py [args]  (-h for options)
"""
from scaling_reranked_retrieval.application.analysis.success_at_20 import main

if __name__ == "__main__":
    main()
