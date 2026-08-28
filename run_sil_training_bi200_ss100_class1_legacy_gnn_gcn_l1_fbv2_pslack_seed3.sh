#!/bin/bash
# 2026-08-09: best confirmed GNN config (GCN L1, lr=0.001, hidden_dim=128,
# dropout=0.0) + feature_builder v2 with the new cr_pickup_slack feature
# (ENABLE_PICKUP_SLACK_FEATURE=True in feat_builder_new.py, on top of the
# already-enabled competition features, 95-dim total). Compare against
# run_sil_training_bi200_ss100_class1_legacy_gnn_gcn_l1_fbv2_seed3.sh (same
# config, pre-pickup_slack code, 94-dim) to see if the feature helps.
set -eu

echo "=== [$(date +%H:%M:%S)] SIL bi200/ss100 class1 legacy - GNN GCN 1L fbv2+pslack seed3 ==="
./venv/bin/python3 rtv_solver/main.py \
  --mode coaml \
  --input_dir "solutions/li_lim/manifests/" \
  --batch_interval 200 --step_size 100 --max_cardinality 2 \
  --learning_rate 0.001 --epochs 5 \
  --hidden_dim 128 --dropout 0.0 \
  --imitation_scoring_rule legacy \
  --use_request_pruner False --use_request_graph_pruner False \
  --model_type gnn --gnn_num_message_passing_layers 1 --gnn_aggregator gcn \
  --feature_builder_version v2 \
  --seed 3 \
  --output_dir "outputs/sil_training_bi200_ss100_class1_legacy_gnn_gcn_l1_fbv2_pslack_seed3" \
  > "sil_training_bi200_ss100_class1_legacy_gnn_gcn_l1_fbv2_pslack_seed3.log" 2>&1
echo "=== [$(date +%H:%M:%S)] DONE: GCN 1L fbv2+pslack seed3 (exit $?) ==="
