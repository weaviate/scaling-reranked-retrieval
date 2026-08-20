#!/usr/bin/env python3
"""Routing-vs-blending oracle decomposition.

Run script — a thin wrapper; all logic lives in src.application.analysis.oracle_config.
Usage: uv run python scripts/oracle_config.py [args]  (-h for options)
"""
from src.application.analysis.oracle_config import main

if __name__ == "__main__":
    main()
