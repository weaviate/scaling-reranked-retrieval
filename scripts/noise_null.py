#!/usr/bin/env python3
"""Noise-null robustness check for the routing claim (winner's-curse null).

Run script — a thin wrapper; all logic lives in
src.application.analysis.noise_null.
Usage: uv run python scripts/noise_null.py [args]  (-h for options)

The full (fusion) sweep is only bit-reproducible under a fixed string hash
seed: the RSF path iterates set(pool), so tie-breaking at the rank-K boundary
depends on PYTHONHASHSEED. Pin it BEFORE the interpreter loads any module by
re-exec'ing — skipped in --singleton-only mode, which never touches fusion
and is tie-free.
"""
import os
import sys

if (
    "--singleton-only" not in sys.argv
    and os.environ.get("PYTHONHASHSEED") != "0"
):
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable, *sys.argv])

from src.application.analysis.noise_null import main  # noqa: E402

if __name__ == "__main__":
    main()
