"""Application layer — the experiment's use cases, orchestrating domain + adapters.

Split by cost: experiments/ spend API calls; analysis/ is zero-network.
Entry points live in scripts/; nothing here parses sys.argv at import time.
"""
