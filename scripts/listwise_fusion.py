#!/usr/bin/env python3
"""RRF fusion analysis over cached listwise rankings.

Run script — a thin wrapper; all logic lives in src.application.analysis.listwise_fusion.
Usage: uv run python scripts/listwise_fusion.py [args]  (-h for options)
"""
from src.application.analysis.listwise_fusion import main

if __name__ == "__main__":
    main()
