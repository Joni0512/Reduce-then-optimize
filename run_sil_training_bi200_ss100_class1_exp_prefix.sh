#!/bin/bash
# 2026-07-27: first "real" cross-file SIL training run with the exponential_prefix
# scoring rule + the y*-ILP global-uniqueness fix + best-val-checkpoint selection
# (see training_loop.py). Class-1 only (standard TRAINING_FILES/VALIDATION_FILES,
# 23 train / 6 val - no override needed), bi200/ss100, mc2, 5 epochs, no pruner
# (isolating the scoring-rule/target-construction change from the pruner).
set -u

echo "=== [$(date +%H:%M:%S)] SIL bi200/ss100 class1 exponential_prefix ==="
./venv/bin/python3 rtv_solver/main.py \
  --mode coaml \
  --input_dir "solutions/li_lim/manifests/" \
  --batch_interval 200 --step_size 100 --max_cardinality 2 \
  --learning_rate 0.0001 --epochs 5 \
  --imitation_scoring_rule exponential_prefix \
  --use_request_pruner False --use_request_graph_pruner False \
  --seed 42 \
  --output_dir "outputs/sil_training_bi200_ss100_class1_exp_prefix" \
  > "sil_training_bi200_ss100_class1_exp_prefix.log" 2>&1
echo "=== [$(date +%H:%M:%S)] DONE: class1 exponential_prefix (exit $?) ==="
