#!/bin/bash
# 2026-08-29: best confirmed GNN config (GCN L1, lr=0.001, hidden_dim=128,
# dropout=0.0) + feature_builder v1 (default) with the new cr_pickup_slack
# feature (ENABLE_PICKUP_SLACK_FEATURE=True in feat_builder.py, 85-dim total
# vs. 84 without). Compare against the original v1 grid-best 5-seed
# confirmation (77.23% +/- 0.57%, pre-pickup_slack code, 84-dim) to see if
# the feature helps on top of v1 too, not just v2+competition.
set -eu

echo "=== [$(date +%H:%M:%S)] SIL bi200/ss100 class1 legacy - GNN GCN 1L v1+pslack seed1 ==="
./venv/bin/python3 rtv_solver/main.py \
  --mode coaml \
  --input_dir "solutions/li_lim/manifests/" \
  --batch_interval 200 --step_size 100 --max_cardinality 2 \
  --learning_rate 0.001 --epochs 5 \
  --hidden_dim 128 --dropout 0.0 \
  --imitation_scoring_rule legacy \
  --use_request_pruner False --use_request_graph_pruner False \
  --model_type gnn --gnn_num_message_passing_layers 1 --gnn_aggregator gcn \
  --feature_builder_version v1 \
  --seed 1 \
  --output_dir "outputs/sil_training_bi200_ss100_class1_legacy_gnn_gcn_l1_v1_pslack_seed1" \
  > "sil_training_bi200_ss100_class1_legacy_gnn_gcn_l1_v1_pslack_seed1.log" 2>&1
echo "=== [$(date +%H:%M:%S)] DONE: GCN 1L v1+pslack seed1 (exit $?) ==="
