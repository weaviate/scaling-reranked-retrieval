"""Domain layer — the pure logic of the experiment.

No network, no filesystem, no provider SDKs. Everything here is deterministic
math and data over plain Python values:

    fusion.py       RRF / RSF fusion over per-provider score dicts, plus the
                    rank-list fusion used at the listwise tier (the RSF
                    set(pool) tie-break semantics documented in its docstring
                    are a pinned behavior contract)
    conditions.py   the experiment condition menu (equal-weight only) and its
                    generator
    metrics.py      metric-name helpers and the extra-recall-cutoff builder
    aggregate.py    mean-across-queries / median-across-subsets aggregation

Domain modules may import src.config (shared constants/paths registry) and
each other — never src.application, src.adapters, or any external service SDK.
"""
