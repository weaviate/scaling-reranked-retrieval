#!/usr/bin/env python3
"""Binary-selection oracle routing at the listwise tier.

Run script — a thin wrapper; all logic lives in src.application.analysis.listwise_oracle_routing.
Usage: uv run python scripts/listwise_oracle_routing.py [args]  (-h for options)
"""
from src.application.analysis.listwise_oracle_routing import main

if __name__ == "__main__":
    main()
