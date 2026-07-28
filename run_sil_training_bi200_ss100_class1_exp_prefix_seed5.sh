#!/bin/bash
# 2026-07-28: seed-5 counterpart to run_sil_training_bi200_ss100_class1_exp_prefix.sh
# for a multi-seed comparison of exponential_prefix at cardinality 2.
set -eu

echo "=== [$(date +%H:%M:%S)] SIL bi200/ss100 class1 exponential_prefix seed5 ==="
./venv/bin/python3 rtv_solver/main.py \
  --mode coaml \
  --input_dir "solutions/li_lim/manifests/" \
  --batch_interval 200 --step_size 100 --max_cardinality 2 \
  --learning_rate 0.0001 --epochs 5 \
  --imitation_scoring_rule exponential_prefix \
  --use_request_pruner False --use_request_graph_pruner False \
  --seed 5 \
  --output_dir "outputs/sil_training_bi200_ss100_class1_exp_prefix_seed5" \
  > "sil_training_bi200_ss100_class1_exp_prefix_seed5.log" 2>&1
echo "=== [$(date +%H:%M:%S)] DONE: class1 exp_prefix bi200/ss100 seed5 (exit $?) ==="
