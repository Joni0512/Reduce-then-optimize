#!/bin/bash
# 2026-08-06: feature-builder v1/v2 comparison, v2 side - identical to
# run_sil_training_bi200_ss100_class1_legacy_fbv1.sh except
# --feature_builder_version v2 (feat_builder_new.py: global future-demand grid
# + candidate request-location features, 87-dim vs. v1's 84-dim), so the two
# scripts differ in exactly that one flag.
set -eu

echo "=== [$(date +%H:%M:%S)] SIL bi200/ss100 class1 legacy - feature_builder v2 ==="
./venv/bin/python3 rtv_solver/main.py \
  --mode coaml \
  --input_dir "solutions/li_lim/manifests/" \
  --batch_interval 200 --step_size 100 --max_cardinality 2 \
  --learning_rate 0.0001 --epochs 5 \
  --imitation_scoring_rule legacy \
  --use_request_pruner False --use_request_graph_pruner False \
  --feature_builder_version v2 \
  --seed 42 \
  --output_dir "outputs/sil_training_bi200_ss100_class1_legacy_fbv2" \
  > "sil_training_bi200_ss100_class1_legacy_fbv2.log" 2>&1
echo "=== [$(date +%H:%M:%S)] DONE: class1 legacy feature_builder v2 (exit $?) ==="
