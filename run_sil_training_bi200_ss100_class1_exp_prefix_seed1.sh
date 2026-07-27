#!/bin/bash
# 2026-07-27: seed-1 counterpart to run_sil_training_bi200_ss100_class1_exp_prefix.sh
# (same class-1/mc2/5-epoch/no-pruner/bi200/ss100 setup, --seed 1 instead of 42)
# for a second-seed comparison. exponential_prefix goes through the global ILP
# y* construction (coaml_pipeline._build_y_star_from_imitation_scores).
set -eu

echo "=== [$(date +%H:%M:%S)] SIL bi200/ss100 class1 exponential_prefix seed1 ==="
./venv/bin/python3 rtv_solver/main.py \
  --mode coaml \
  --input_dir "solutions/li_lim/manifests/" \
  --batch_interval 200 --step_size 100 --max_cardinality 2 \
  --learning_rate 0.0001 --epochs 5 \
  --imitation_scoring_rule exponential_prefix \
  --use_request_pruner False --use_request_graph_pruner False \
  --seed 1 \
  --output_dir "outputs/sil_training_bi200_ss100_class1_exp_prefix_seed1" \
  > "sil_training_bi200_ss100_class1_exp_prefix_seed1.log" 2>&1
echo "=== [$(date +%H:%M:%S)] DONE: class1 exp_prefix bi200/ss100 seed1 (exit $?) ==="
