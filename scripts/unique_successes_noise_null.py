#!/usr/bin/env python3
"""Noise-null twin of the unique-success analysis.

Run script — a thin wrapper; all logic lives in src.application.analysis.unique_successes_noise_null.
Usage: uv run python scripts/unique_successes_noise_null.py [args]  (-h for options)
"""
from src.application.analysis.unique_successes_noise_null import main

if __name__ == "__main__":
    main()
