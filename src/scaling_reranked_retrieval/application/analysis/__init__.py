"""Zero-network analysis services.

Everything here reads results/ artifacts (caches, run summaries, cached
listwise rankings) and writes derived reports — no reranker, retrieval, or
LLM API is ever called. Each module is a self-contained use case with a
main(); run them via the same-named wrappers in scripts/.

CE (cross-encoder) tier: agreement, oracle_config, unique_successes,
unique_successes_noise_null, equal_weight, noise_null, singleton_deep_recall,
success_at_20. Listwise tier: listwise_fusion, listwise_unique_successes,
listwise_oracle_routing, listwise_self_oracle.
"""
