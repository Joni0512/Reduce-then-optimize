#!/bin/bash
# 2026-08-02: chain-submit the 4 ablation-stage-2 (concat combine) screening
# runs for seeds 2 and 3 (1 and 2 message-passing layers each). Runs alongside
# the local seed 42/1 concat screens on this machine - together 4 of the
# eventual 6 seeds per 1L/2L-concat combination. Sequential (afterany) to
# avoid the concurrent-Gurobi-session license overage seen before.
set -eu

already_running_id="${1:-}"

JOBS=(
  submit_sil_bi200_ss100_class1_legacy_gnn_concat_l1_seed2.sbatch
  submit_sil_bi200_ss100_class1_legacy_gnn_concat_l1_seed3.sbatch
  submit_sil_bi200_ss100_class1_legacy_gnn_concat_l2_seed2.sbatch
  submit_sil_bi200_ss100_class1_legacy_gnn_concat_l2_seed3.sbatch
)

prev_id="$already_running_id"
if [ -n "$prev_id" ]; then
  echo "Chaining after already-submitted job: $prev_id"
fi
for job in "${JOBS[@]}"; do
  if [ -z "$prev_id" ]; then
    raw_id=$(sbatch -M serial --parsable "$job")
  else
    raw_id=$(sbatch -M serial --parsable --dependency=afterany:"$prev_id" "$job")
  fi
  id="${raw_id%%;*}"
  echo "Submitted $job as job $id (depends on: ${prev_id:-none})"
  prev_id="$id"
done

echo "Chain complete. Last job: $prev_id"
echo "Check status any time with: squeue -M serial -u \$USER"
