#!/bin/bash
# 2026-07-27: legacy-scoring-rule counterpart to
# run_sil_training_bi200_ss100_class1_exp_prefix.sh - same instances (class-1,
# standard TRAINING_FILES/VALIDATION_FILES), same bi200/ss100/mc2/5-epoch/no-pruner
# setup, only --imitation_scoring_rule differs (legacy vs exponential_prefix), for
# a direct scoring-rule ablation with everything else held constant.
set -u

echo "=== [$(date +%H:%M:%S)] SIL bi200/ss100 class1 legacy ==="
./venv/bin/python3 rtv_solver/main.py \
  --mode coaml \
  --input_dir "solutions/li_lim/manifests/" \
  --batch_interval 200 --step_size 100 --max_cardinality 2 \
  --learning_rate 0.0001 --epochs 5 \
  --imitation_scoring_rule legacy \
  --use_request_pruner False --use_request_graph_pruner False \
  --seed 42 \
  --output_dir "outputs/sil_training_bi200_ss100_class1_legacy" \
  > "sil_training_bi200_ss100_class1_legacy.log" 2>&1
echo "=== [$(date +%H:%M:%S)] DONE: class1 legacy (exit $?) ==="
