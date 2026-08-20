#!/usr/bin/env python3
"""Equal-weight 3-way vs best equal-weight pair comparison.

Run script — a thin wrapper; all logic lives in src.application.analysis.equal_weight.
Usage: uv run python scripts/equal_weight.py [args]  (-h for options)
"""
from src.application.analysis.equal_weight import main

if __name__ == "__main__":
    main()
