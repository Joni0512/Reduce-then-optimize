#!/bin/bash
# 2026-08-07: seed-7 counterpart to run_sil_training_bi200_ss100_class1_legacy_fbv1.sh,
# expanding the v1 sample for the v1/v2 feature-builder comparison.
set -eu

echo "=== [$(date +%H:%M:%S)] SIL bi200/ss100 class1 legacy - feature_builder v1 seed7 ==="
./venv/bin/python3 rtv_solver/main.py \
  --mode coaml \
  --input_dir "solutions/li_lim/manifests/" \
  --batch_interval 200 --step_size 100 --max_cardinality 2 \
  --learning_rate 0.0001 --epochs 5 \
  --imitation_scoring_rule legacy \
  --use_request_pruner False --use_request_graph_pruner False \
  --feature_builder_version v1 \
  --seed 7 \
  --output_dir "outputs/sil_training_bi200_ss100_class1_legacy_fbv1_seed7" \
  > "sil_training_bi200_ss100_class1_legacy_fbv1_seed7.log" 2>&1
echo "=== [$(date +%H:%M:%S)] DONE: class1 legacy feature_builder v1 seed7 (exit $?) ==="
