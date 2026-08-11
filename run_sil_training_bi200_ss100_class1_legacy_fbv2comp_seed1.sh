#!/bin/bash
# 2026-08-09: v2 + competition features (ENABLE_COMPETITION_FEATURES=True in
# feat_builder_new.py, second wave of v2 additions - cf_cost_minus_best_for_vehicle
# etc., see CompetitionFeatures). Separate output/log names from the existing
# fbv2_seed*.sh runs (which used v2 WITHOUT competition features, 87-dim) so
# neither overwrites the other - lets us compare baseline (v1) vs v2 vs
# v2+competition (94-dim) side by side.
set -eu

echo "=== [$(date +%H:%M:%S)] SIL bi200/ss100 class1 legacy - feature_builder v2+competition seed1 ==="
./venv/bin/python3 rtv_solver/main.py \
  --mode coaml \
  --input_dir "solutions/li_lim/manifests/" \
  --batch_interval 200 --step_size 100 --max_cardinality 2 \
  --learning_rate 0.0001 --epochs 5 \
  --imitation_scoring_rule legacy \
  --use_request_pruner False --use_request_graph_pruner False \
  --feature_builder_version v2 \
  --seed 1 \
  --output_dir "outputs/sil_training_bi200_ss100_class1_legacy_fbv2comp_seed1" \
  > "sil_training_bi200_ss100_class1_legacy_fbv2comp_seed1.log" 2>&1
echo "=== [$(date +%H:%M:%S)] DONE: class1 legacy feature_builder v2+competition seed1 (exit $?) ==="
