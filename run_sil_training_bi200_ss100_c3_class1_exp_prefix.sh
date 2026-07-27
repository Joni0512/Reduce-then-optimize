#!/bin/bash
# 2026-07-27: cardinality-3 counterpart to
# run_sil_training_bi200_ss100_class1_exp_prefix.sh - same class-1/5-epoch/
# no-pruner/bi200/ss100 setup, --max_cardinality 3 instead of 2. Expect this to
# run much longer than the cardinality-2 version: the C3 single-instance batch
# already showed several class-2 instances blowing up combinatorially (some
# hit a 30min per-instance cap) - a full 23-file training epoch at cardinality 3
# could take substantially longer per epoch than at cardinality 2.
set -eu

echo "=== [$(date +%H:%M:%S)] SIL bi200/ss100 c3 class1 exponential_prefix ==="
./venv/bin/python3 rtv_solver/main.py \
  --mode coaml \
  --input_dir "solutions/li_lim/manifests/" \
  --batch_interval 200 --step_size 100 --max_cardinality 3 \
  --learning_rate 0.0001 --epochs 5 \
  --imitation_scoring_rule exponential_prefix \
  --use_request_pruner False --use_request_graph_pruner False \
  --seed 42 \
  --output_dir "outputs/sil_training_bi200_ss100_c3_class1_exp_prefix" \
  > "sil_training_bi200_ss100_c3_class1_exp_prefix.log" 2>&1
echo "=== [$(date +%H:%M:%S)] DONE: c3 class1 exp_prefix bi200/ss100 (exit $?) ==="
