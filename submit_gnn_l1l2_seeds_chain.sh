#!/bin/bash
# 2026-08-01: chain-submit the 8 GNN 1-vs-2-message-passing-layer seed-variant
# runs (seeds 1-4, each with 1 and 2 layers - seed 42 for both already ran
# separately). Sequential (afterany) to avoid the concurrent-Gurobi-session
# license overage that hit submit_remaining_chain.sh's runs earlier. Run this
# once; SLURM handles the rest unattended.
set -eu

# 2026-08-01: takes an optional first arg - a job ID already submitted (e.g. from
# a previous partial run of this script) to chain the first pending job after,
# instead of submitting it fresh. Leave empty for a normal full run.
already_running_id="${1:-}"

JOBS=(
  submit_sil_bi200_ss100_class1_legacy_gnn_l1_seed1.sbatch
  submit_sil_bi200_ss100_class1_legacy_gnn_l1_seed2.sbatch
  submit_sil_bi200_ss100_class1_legacy_gnn_l1_seed3.sbatch
  submit_sil_bi200_ss100_class1_legacy_gnn_l1_seed4.sbatch
  submit_sil_bi200_ss100_class1_legacy_gnn_l2_seed1.sbatch
  submit_sil_bi200_ss100_class1_legacy_gnn_l2_seed2.sbatch
  submit_sil_bi200_ss100_class1_legacy_gnn_l2_seed3.sbatch
  submit_sil_bi200_ss100_class1_legacy_gnn_l2_seed4.sbatch
)

# sbatch --parsable -M serial returns "jobid;cluster" (e.g. "5344211;serial") -
# strip everything from the first ";" so --dependency=afterany:<id> gets a
# plain job ID, not the raw "jobid;cluster" string (see submit_remaining_chain.sh,
# which hit "Job dependency problem" the first time this bug was present).
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
