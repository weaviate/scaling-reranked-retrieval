#!/usr/bin/env bash
# Re-derive the CE depth-sweep runs for the two refreshed subsets, matching the
# June layout exactly (checklist item 9 in updated_paper_8_28.md):
#
#   results/<subdir>/runs/k{N}_from_k2000.json        reranked_k=20 (Table 1,
#                                                     fusion table, appendix
#                                                     singleton tables)
#   results/<subdir>/runs_rk100/k{N}_from_k2000.json  reranked_k=100 (carries
#                                                     recall_at_50/100 for the
#                                                     capture section; there is
#                                                     no separate rk50 sweep)
#
# All derived (--from-cache) runs: ZERO reranker/LLM API calls, everything
# comes from the refreshed k2000 caches. Also copies the June runs for the
# three unchanged subsets from the backup (read-only source) so cross-subset
# means can be recomputed over all five.
set -euo pipefail
cd "$(dirname "$0")/.."

BACKUP=backup-scaling-reranked-retrieval/local-result-backup/results

for subdir in bright_biology bright_economics bright_psychology; do
  for runs in runs runs_rk100; do
    src="${BACKUP}/${subdir}/${runs}"
    if [ ! -d "${src}" ]; then
      # bright_psychology has no runs/ in the backup — its rk20 metrics were
      # always sourced from runs_rk100 (S@1/nDCG@10/R@20 are prefix metrics,
      # identical under either output cap).
      echo "  (no ${runs}/ in backup for ${subdir} — skipping)"
      continue
    fi
    mkdir -p "results/${subdir}/${runs}"
    # biology's runs/ also holds legacy k*_from_k500.json files — the paper's
    # per-family biology S@1/R@20 sourcing uses them (nDCG is canonical from
    # k2000), so copy everything, not just the k2000-derived files.
    cp "${src}/"k*_from_k*.json "results/${subdir}/${runs}/"
  done
  echo "copied June runs (unchanged subset): ${subdir}"
done

for dataset in earth_science robotics; do
  echo
  echo "=============================================================="
  echo "== k sweep (reranked_k=20): ${dataset}"
  echo "=============================================================="
  uv run python scripts/k_sweep.py "${dataset}"

  echo
  echo "=============================================================="
  echo "== k sweep (reranked_k=100, R@50/R@100): ${dataset}"
  echo "=============================================================="
  uv run python scripts/k_sweep.py "${dataset}" --reranked-k 100
done

echo
echo "Done. Derived runs under results/bright_{earth_science,robotics}/{runs,runs_rk100}/"
echo "plus June copies for biology/economics/psychology."
