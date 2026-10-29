#!/bin/bash
# 2026-10-29: stage 1 of the pruner window study: RHO-no-learning baselines (cardinality 2) for the windows batch_interval/step_size = 40/10, 100/25, 120/30, 200/50,
# each split into 2 shards -> 8 jobs x 4 CPUs on serial_std. Labels for the per-window request pruners (stage 2).
set -eu
cd "$(dirname "$0")"
for w in "40 10" "100 25" "120 30" "200 50"; do
  set -- $w
  for shard in 0 1; do
    sbatch --job-name=rh_base_$1_$2 --export=ALL,BI=$1,SS=$2,SHARD=$shard,NSHARDS=2 submit_baseline_data_window.sbatch
  done
done
