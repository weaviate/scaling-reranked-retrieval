#!/usr/bin/env python3
"""Listwise-LLM reranking over a CE-selected pool (LIVE OpenAI calls).

Run script — a thin wrapper; all logic lives in src.application.experiments.listwise_rerank.
Usage: uv run python scripts/listwise_rerank.py [args]  (-h for options)
"""
from src.application.experiments.listwise_rerank import main

if __name__ == "__main__":
    main()
