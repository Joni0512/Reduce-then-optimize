#!/bin/bash
# 2026-08-04: HP-tuning stage 2 screening run - GCN L1, second batch with
# dropout=0.1 fixed (lr x hidden_dim x dropout grid, 18 combos total).
set -eu

echo "=== [$(date +%H:%M:%S)] SIL bi200/ss100 class1 legacy - GNN GCN 1L HP lr=0.0001 h=256 do=0.1 seed1 ==="
./venv/bin/python3 rtv_solver/main.py \
  --mode coaml \
  --input_dir "solutions/li_lim/manifests/" \
  --batch_interval 200 --step_size 100 --max_cardinality 2 \
  --learning_rate 0.0001 --epochs 5 \
  --hidden_dim 256 --dropout 0.1 \
  --imitation_scoring_rule legacy \
  --use_request_pruner False --use_request_graph_pruner False \
  --model_type gnn --gnn_num_message_passing_layers 1 --gnn_aggregator gcn \
  --seed 1 \
  --output_dir "outputs/sil_training_bi200_ss100_class1_legacy_gnn_gcn_l1_hp_lr1e4_h256_d1_seed1" \
  > "sil_training_bi200_ss100_class1_legacy_gnn_gcn_l1_hp_lr1e4_h256_d1_seed1.log" 2>&1
echo "=== [$(date +%H:%M:%S)] DONE: GCN 1L HP lr=0.0001 h=256 do=0.1 seed1 (exit $?) ==="
