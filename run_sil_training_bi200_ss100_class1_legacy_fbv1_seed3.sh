#!/bin/bash
# 2026-08-08: seed-3 counterpart to run_sil_training_bi200_ss100_class1_legacy_fbv1.sh,
# expanding the v1 sample for the v1/v2 feature-builder comparison (v2 already has
# seeds 1,2,3,4,5,6,7,42 - filling in the matching v1 seeds).
set -eu

echo "=== [$(date +%H:%M:%S)] SIL bi200/ss100 class1 legacy - feature_builder v1 seed3 ==="
./venv/bin/python3 rtv_solver/main.py \
  --mode coaml \
  --input_dir "solutions/li_lim/manifests/" \
  --batch_interval 200 --step_size 100 --max_cardinality 2 \
  --learning_rate 0.0001 --epochs 5 \
  --imitation_scoring_rule legacy \
  --use_request_pruner False --use_request_graph_pruner False \
  --feature_builder_version v1 \
  --seed 3 \
  --output_dir "outputs/sil_training_bi200_ss100_class1_legacy_fbv1_seed3" \
  > "sil_training_bi200_ss100_class1_legacy_fbv1_seed3.log" 2>&1
echo "=== [$(date +%H:%M:%S)] DONE: class1 legacy feature_builder v1 seed3 (exit $?) ==="
