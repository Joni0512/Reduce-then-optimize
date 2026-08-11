#!/bin/bash
# 2026-08-08: best confirmed GNN config (GCN L1, lr=0.001, hidden_dim=128,
# dropout=0.0 - 5-seed mean 77.23% with feature_builder v1) combined with
# feature_builder_version=v2 (feat_builder_new.py) to see if the newer
# feature set helps the best-known architecture, not just the MLP baseline.
set -eu

echo "=== [$(date +%H:%M:%S)] SIL bi200/ss100 class1 legacy - GNN GCN 1L fbv2 seed1 ==="
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
  --seed 1 \
  --output_dir "outputs/sil_training_bi200_ss100_class1_legacy_gnn_gcn_l1_fbv2_seed1" \
  > "sil_training_bi200_ss100_class1_legacy_gnn_gcn_l1_fbv2_seed1.log" 2>&1
echo "=== [$(date +%H:%M:%S)] DONE: GCN 1L fbv2 seed1 (exit $?) ==="
