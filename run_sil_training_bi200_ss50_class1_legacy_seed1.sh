#!/bin/bash
# 2026-07-27: seed-1 counterpart to run_sil_training_bi200_ss50_class1_legacy.sh
# (same class-1/mc2/5-epoch/no-pruner/bi200/ss50 setup, --seed 1 instead of 42).
# legacy goes through the cheap per-vehicle y* selection (no global ILP needed).
set -eu

echo "=== [$(date +%H:%M:%S)] SIL bi200/ss50 class1 legacy seed1 ==="
./venv/bin/python3 rtv_solver/main.py \
  --mode coaml \
  --input_dir "solutions/li_lim/manifests/" \
  --batch_interval 200 --step_size 50 --max_cardinality 2 \
  --learning_rate 0.0001 --epochs 5 \
  --imitation_scoring_rule legacy \
  --use_request_pruner False --use_request_graph_pruner False \
  --seed 1 \
  --output_dir "outputs/sil_training_bi200_ss50_class1_legacy_seed1" \
  > "sil_training_bi200_ss50_class1_legacy_seed1.log" 2>&1
echo "=== [$(date +%H:%M:%S)] DONE: class1 legacy bi200/ss50 seed1 (exit $?) ==="
