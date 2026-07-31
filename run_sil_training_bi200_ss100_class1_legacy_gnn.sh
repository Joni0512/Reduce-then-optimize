#!/bin/bash
# 2026-07-30: GNN counterpart to run_sil_training_bi200_ss100_class1_legacy.sh -
# identical setup (class-1, standard TRAINING_FILES/VALIDATION_FILES, bi200/ss100/mc2/
# 5-epoch/no-pruner, legacy scoring rule, seed 42), only --model_type differs
# (gnn = CandidateScoringGNN, conflict-graph message passing, vs the MLP baseline).
# Uses the CURRENT feat_builder.py features (not yet feat_builder_new.py), so this
# is a like-for-like architecture comparison against the existing MLP checkpoint at
# outputs/sil_training_bi200_ss100_class1_legacy/ - same features, same everything
# else, only the scoring model differs.
set -eu

echo "=== [$(date +%H:%M:%S)] SIL bi200/ss100 class1 legacy - GNN ==="
./venv/bin/python3 rtv_solver/main.py \
  --mode coaml \
  --input_dir "solutions/li_lim/manifests/" \
  --batch_interval 200 --step_size 100 --max_cardinality 2 \
  --learning_rate 0.0001 --epochs 5 \
  --imitation_scoring_rule legacy \
  --use_request_pruner False --use_request_graph_pruner False \
  --model_type gnn --gnn_num_message_passing_layers 1 \
  --seed 42 \
  --output_dir "outputs/sil_training_bi200_ss100_class1_legacy_gnn" \
  > "sil_training_bi200_ss100_class1_legacy_gnn.log" 2>&1
echo "=== [$(date +%H:%M:%S)] DONE: class1 legacy GNN (exit $?) ==="
