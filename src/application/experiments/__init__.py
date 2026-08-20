"""Live-call experiment services (these SPEND — API calls or wall-clock).

    run_experiment.py       the main harness: collect / derive / real-call
                            modes over the condition menu
    hybrid_variance.py      across-trial variance of first-stage hybrid
                            retrieval (the single-trial justification)
    score_variance.py       reranker determinism check
    latency_measurement.py  per-provider rerank latency vs K + hybrid
                            latency/payload
    listwise_rerank.py      listwise-LLM reranking over a CE-selected pool
                            (the only module that calls OpenAI)
    listwise_top100.py      the top-100 listwise conditions harness

Run these via their wrappers in scripts/ — they need API keys (see
src.application.cli for the guards).
"""
