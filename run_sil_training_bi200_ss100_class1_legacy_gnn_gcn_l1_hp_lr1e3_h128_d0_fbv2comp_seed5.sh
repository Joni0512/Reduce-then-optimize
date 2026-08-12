#!/bin/bash
# 2026-08-12: seed-5 counterpart to run_sil_training_bi200_ss100_class1_legacy_gnn_gcn_l1_hp_lr1e3_h128_d0_fbv2comp_seed1.sh
# (v2 + competition features + GNN, see that script's header comment for context).
set -eu

echo "=== [$(date +%H:%M:%S)] SIL bi200/ss100 class1 legacy - GNN GCN 1L HP lr=0.001 h=128 do=0.0 fbv2comp seed5 ==="
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
  --seed 5 \
  --output_dir "outputs/sil_training_bi200_ss100_class1_legacy_gnn_gcn_l1_hp_lr1e3_h128_d0_fbv2comp_seed5" \
  > "sil_training_bi200_ss100_class1_legacy_gnn_gcn_l1_hp_lr1e3_h128_d0_fbv2comp_seed5.log" 2>&1
echo "=== [$(date +%H:%M:%S)] DONE: GCN 1L HP lr=0.001 h=128 do=0.0 fbv2comp seed5 (exit $?) ==="
