#!/bin/bash
# 2026-07-27: same class-1 / mc2 / 5-epoch / no-pruner setup as
# run_sil_training_bi200_ss100_class1_exp_prefix.sh, but bi100/ss50 (tighter
# rolling horizon window) instead of bi200/ss100.
set -eu

echo "=== [$(date +%H:%M:%S)] SIL bi100/ss50 class1 exponential_prefix ==="
./venv/bin/python3 rtv_solver/main.py \
  --mode coaml \
  --input_dir "solutions/li_lim/manifests/" \
  --batch_interval 100 --step_size 50 --max_cardinality 2 \
  --learning_rate 0.0001 --epochs 5 \
  --imitation_scoring_rule exponential_prefix \
  --use_request_pruner False --use_request_graph_pruner False \
  --seed 42 \
  --output_dir "outputs/sil_training_bi100_ss50_class1_exp_prefix" \
  > "sil_training_bi100_ss50_class1_exp_prefix.log" 2>&1
echo "=== [$(date +%H:%M:%S)] DONE: class1 exp_prefix bi100/ss50 (exit $?) ==="
