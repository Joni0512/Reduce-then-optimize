#!/bin/bash
# 2026-07-27: legacy-scoring-rule counterpart to
# run_sil_training_bi100_ss50_class1_exp_prefix.sh - same class-1/mc2/5-epoch/
# no-pruner/bi100/ss50 setup, only --imitation_scoring_rule differs.
set -eu

echo "=== [$(date +%H:%M:%S)] SIL bi100/ss50 class1 legacy ==="
./venv/bin/python3 rtv_solver/main.py \
  --mode coaml \
  --input_dir "solutions/li_lim/manifests/" \
  --batch_interval 100 --step_size 50 --max_cardinality 2 \
  --learning_rate 0.0001 --epochs 5 \
  --imitation_scoring_rule legacy \
  --use_request_pruner False --use_request_graph_pruner False \
  --seed 42 \
  --output_dir "outputs/sil_training_bi100_ss50_class1_legacy" \
  > "sil_training_bi100_ss50_class1_legacy.log" 2>&1
echo "=== [$(date +%H:%M:%S)] DONE: class1 legacy bi100/ss50 (exit $?) ==="
