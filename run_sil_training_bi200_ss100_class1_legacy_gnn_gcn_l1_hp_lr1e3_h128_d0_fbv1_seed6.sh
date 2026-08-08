#!/bin/bash
# 2026-08-08: seed-6 extension of the confirmed-best GCN L1 combo (lr=0.001,
# h=128, dropout=0) with v1 feature-builder made explicit, pairing with the v2
# seed-6 counterpart for the GNN v1/v2 feature-builder comparison (extending
# from 5 to 7 seeds, matching the MLP comparison's seed set).
set -eu

echo "=== [$(date +%H:%M:%S)] SIL bi200/ss100 class1 legacy - GNN GCN 1L HP lr=0.001 h=128 do=0.0 fbv1 seed6 ==="
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
  --seed 6 \
  --output_dir "outputs/sil_training_bi200_ss100_class1_legacy_gnn_gcn_l1_hp_lr1e3_h128_d0_fbv1_seed6" \
  > "sil_training_bi200_ss100_class1_legacy_gnn_gcn_l1_hp_lr1e3_h128_d0_fbv1_seed6.log" 2>&1
echo "=== [$(date +%H:%M:%S)] DONE: GCN 1L HP lr=0.001 h=128 do=0.0 fbv1 seed6 (exit $?) ==="
