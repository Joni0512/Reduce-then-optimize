#!/bin/bash
# 2026-08-29: bi400/ss100 counterpart to the bi200/ss100 GCN L1 v1+pslack best
# result (78.93% +/- 0.32%, lr=0.001, hidden_dim=128, dropout=0.0). Tests
# whether cr_pickup_slack still helps at a larger batch_interval. Explicit
# --enable_pickup_slack_feature True (matches class default, stated for
# clarity/reproducibility alongside the seed*_seed*.sh no-pslack baseline).
set -eu

echo "=== [$(date +%H:%M:%S)] SIL bi400/ss100 class1 legacy - GNN GCN 1L v1+pslack seed4 ==="
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
  --enable_pickup_slack_feature True \
  --seed 4 \
  --output_dir "outputs/sil_training_bi400_ss100_class1_legacy_gnn_gcn_l1_v1_pslack_seed4" \
  > "sil_training_bi400_ss100_class1_legacy_gnn_gcn_l1_v1_pslack_seed4.log" 2>&1
echo "=== [$(date +%H:%M:%S)] DONE: GCN 1L v1+pslack seed4 (exit $?) ==="
