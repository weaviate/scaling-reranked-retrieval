#!/usr/bin/env python3
"""Main harness: collect / derive / real-call sweeps over the condition menu (LIVE API in collect/real modes).

Run script — a thin wrapper; all logic lives in src.application.experiments.run_experiment.
Usage: uv run python scripts/run_experiment.py [args]  (-h for options)
"""
from src.application.experiments.run_experiment import main

if __name__ == "__main__":
    main()
