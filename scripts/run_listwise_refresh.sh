#!/usr/bin/env bash
# Re-run the listwise experiment trio over the REFRESHED zerank top-20 pools
# for earth_science + robotics (pools rebuilt 2026-08-25 from the refreshed
# k2000 caches; June (query, trial) LLM calls whose pool set is unchanged are
# reused from results/listwise/cache/ — the per-entry doc-set guard re-pays
# only queries whose shortlist actually changed).
#
# Per model: earth_science 90 new calls (258/348 cached), robotics 144 new
# calls (132/276 cached). Pre-flight cost prints before any call.
#
# Usage:  OPENAI_API_KEY must be set.
#   bash scripts/run_listwise_refresh.sh
set -euo pipefail
cd "$(dirname "$0")/.."

: "${OPENAI_API_KEY:?OPENAI_API_KEY is not set}"

for domain in earth_science robotics; do
  for model in gpt-5.4-mini gpt-5.6-luna gpt-5.6-terra; do
    echo
    echo "=============================================================="
    echo "== listwise: ${domain} / ${model} (effort none, zerank_only top-20)"
    echo "=============================================================="
    uv run python scripts/listwise_rerank.py \
      --domain "${domain}" \
      --model "${model}" \
      --reasoning-effort none \
      --pool-source zerank_only \
      --first-stage-k 2000 \
      --pool-k 20 \
      --trials 3
  done
done

echo
echo "All six runs complete. Outputs under results/listwise/runs/zerank-only__first2000__top20/"
