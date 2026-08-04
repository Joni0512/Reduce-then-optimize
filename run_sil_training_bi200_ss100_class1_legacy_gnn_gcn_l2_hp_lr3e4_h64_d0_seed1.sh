#!/bin/bash
# 2026-08-05: HP-tuning stage 2 screening run - GCN L2 (no HP data yet,
# biggest gap in the sweep). Single seed per HP combo for coarse grid
# screening (lr x hidden_dim, dropout=0.0 fixed).
set -eu

echo "=== [$(date +%H:%M:%S)] SIL bi200/ss100 class1 legacy - GNN GCN 2L HP lr=0.0003 h=64 do=0.0 seed1 ==="
./venv/bin/python3 rtv_solver/main.py \
  --mode coaml \
  --input_dir "solutions/li_lim/manifests/" \
  --batch_interval 200 --step_size 100 --max_cardinality 2 \
  --learning_rate 0.0003 --epochs 5 \
  --hidden_dim 64 --dropout 0.0 \
  --imitation_scoring_rule legacy \
  --use_request_pruner False --use_request_graph_pruner False \
  --model_type gnn --gnn_num_message_passing_layers 2 --gnn_aggregator gcn \
  --seed 1 \
  --output_dir "outputs/sil_training_bi200_ss100_class1_legacy_gnn_gcn_l2_hp_lr3e4_h64_d0_seed1" \
  > "sil_training_bi200_ss100_class1_legacy_gnn_gcn_l2_hp_lr3e4_h64_d0_seed1.log" 2>&1
echo "=== [$(date +%H:%M:%S)] DONE: GCN 2L HP lr=0.0003 h=64 do=0.0 seed1 (exit $?) ==="
