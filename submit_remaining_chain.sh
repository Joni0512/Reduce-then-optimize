#!/bin/bash
# 2026-07-28: chain-submit the 8 remaining runs (2 seed1 exp_prefix C2 retries +
# all 6 C3 runs, all previously failed on Gurobi license overage from running
# too many concurrent jobs). Each job only starts once the previous one has
# finished (afterany = regardless of success/failure), so at most one of these
# runs Gurobi at a time - avoids the concurrent-session overage that killed
# almost every job submitted in parallel earlier. Run this once; SLURM handles
# the rest unattended.
set -eu

# 2026-07-28: takes an optional first arg - a job ID already submitted (e.g. from
# a previous partial run of this script) to chain the first pending job after,
# instead of submitting it fresh. Leave empty for a normal full run.
already_running_id="${1:-}"

JOBS=(
  submit_sil_bi200_ss100_class1_exp_prefix_seed1.sbatch
  submit_sil_bi200_ss50_class1_exp_prefix_seed1.sbatch
  submit_sil_bi200_ss100_c3_class1_exp_prefix.sbatch
  submit_sil_bi200_ss50_c3_class1_exp_prefix.sbatch
  submit_sil_bi100_ss50_c3_class1_exp_prefix.sbatch
  submit_sil_bi200_ss100_c3_class1_legacy.sbatch
  submit_sil_bi200_ss50_c3_class1_legacy.sbatch
  submit_sil_bi100_ss50_c3_class1_legacy.sbatch
)

# sbatch --parsable -M serial returns "jobid;cluster" (e.g. "5344211;serial") -
# strip everything from the first ";" so --dependency=afterany:<id> gets a
# plain job ID, not the raw "jobid;cluster" string (which caused "Job
# dependency problem" the first time this ran).
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
