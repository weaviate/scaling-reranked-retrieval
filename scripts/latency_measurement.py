#!/usr/bin/env python3
"""Per-provider rerank latency vs K + hybrid latency/payload (LIVE calls).

Run script — a thin wrapper; all logic lives in src.application.experiments.latency_measurement.
Usage: uv run python scripts/latency_measurement.py [args]  (-h for options)
"""
from src.application.experiments.latency_measurement import main

if __name__ == "__main__":
    main()
