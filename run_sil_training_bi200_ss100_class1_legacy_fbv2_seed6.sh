#!/bin/bash
# 2026-08-07: seed-6 counterpart to run_sil_training_bi200_ss100_class1_legacy_fbv2.sh,
# expanding the v2 sample (with reject-grid fix, see coaml_pipeline.py) for the v1/v2
# feature-builder comparison.
set -eu

echo "=== [$(date +%H:%M:%S)] SIL bi200/ss100 class1 legacy - feature_builder v2 seed6 ==="
./venv/bin/python3 rtv_solver/main.py \
  --mode coaml \
  --input_dir "solutions/li_lim/manifests/" \
  --batch_interval 200 --step_size 100 --max_cardinality 2 \
  --learning_rate 0.0001 --epochs 5 \
  --imitation_scoring_rule legacy \
  --use_request_pruner False --use_request_graph_pruner False \
  --feature_builder_version v2 \
  --seed 6 \
  --output_dir "outputs/sil_training_bi200_ss100_class1_legacy_fbv2_seed6" \
  > "sil_training_bi200_ss100_class1_legacy_fbv2_seed6.log" 2>&1
echo "=== [$(date +%H:%M:%S)] DONE: class1 legacy feature_builder v2 seed6 (exit $?) ==="
