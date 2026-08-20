#!/usr/bin/env python3
"""Self-ensemble oracle control (listwise winner's-curse null).

Run script — a thin wrapper; all logic lives in src.application.analysis.listwise_self_oracle.
Usage: uv run python scripts/listwise_self_oracle.py [args]  (-h for options)
"""
from src.application.analysis.listwise_self_oracle import main

if __name__ == "__main__":
    main()
