#!/usr/bin/env python3
"""Top-100 listwise conditions harness (LIVE OpenAI calls).

Run script — a thin wrapper; all logic lives in src.application.experiments.listwise_top100.
Usage: uv run python scripts/listwise_top100.py [args]  (-h for options)
"""
from src.application.experiments.listwise_top100 import main

if __name__ == "__main__":
    main()
