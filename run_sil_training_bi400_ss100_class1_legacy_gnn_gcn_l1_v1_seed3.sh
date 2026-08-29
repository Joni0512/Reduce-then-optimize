#!/bin/bash
# 2026-08-29: bi400/ss100 counterpart to the bi200/ss100 GCN L1 v1 grid-best
# confirmation (lr=0.001, hidden_dim=128, dropout=0.0). Explicit
# --enable_pickup_slack_feature False baseline (class default is now True) so
# this stays reproducible for the with/without pickup_slack comparison at
# this batch_interval/step_size, using the new CLI override (see
# pipeline.build_feature_builder()) instead of a source-level toggle.
set -eu

echo "=== [$(date +%H:%M:%S)] SIL bi400/ss100 class1 legacy - GNN GCN 1L v1 (no pslack) seed3 ==="
./venv/bin/python3 rtv_solver/main.py \
  --mode coaml \
  --input_dir "solutions/li_lim/manifests/" \
  --batch_interval 400 --step_size 100 --max_cardinality 2 \
  --learning_rate 0.001 --epochs 5 \
  --hidden_dim 128 --dropout 0.0 \
  --imitation_scoring_rule legacy \
  --use_request_pruner False --use_request_graph_pruner False \
  --model_type gnn --gnn_num_message_passing_layers 1 --gnn_aggregator gcn \
  --feature_builder_version v1 \
  --enable_pickup_slack_feature False \
  --seed 3 \
  --output_dir "outputs/sil_training_bi400_ss100_class1_legacy_gnn_gcn_l1_v1_seed3" \
  > "sil_training_bi400_ss100_class1_legacy_gnn_gcn_l1_v1_seed3.log" 2>&1
echo "=== [$(date +%H:%M:%S)] DONE: GCN 1L v1 (no pslack) seed3 (exit $?) ==="
