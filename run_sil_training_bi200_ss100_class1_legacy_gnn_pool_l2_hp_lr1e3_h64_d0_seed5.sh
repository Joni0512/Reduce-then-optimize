#!/bin/bash
# 2026-08-04: HP-tuning stage 2 - Pool L2, lr=0.001, h=64, dropout=0.0,
# seed 5. Testing whether the higher-learning-rate trend (confirmed for
# GCN L1, Mean L1/L2, Pool L1) also holds for Pool L2.
set -eu

echo "=== [$(date +%H:%M:%S)] SIL bi200/ss100 class1 legacy - GNN Pool 2L HP lr=0.001 h=64 do=0.0 seed5 ==="
./venv/bin/python3 rtv_solver/main.py \
  --mode coaml \
  --input_dir "solutions/li_lim/manifests/" \
  --batch_interval 200 --step_size 100 --max_cardinality 2 \
  --learning_rate 0.001 --epochs 5 \
  --hidden_dim 64 --dropout 0.0 \
  --imitation_scoring_rule legacy \
  --use_request_pruner False --use_request_graph_pruner False \
  --model_type gnn --gnn_num_message_passing_layers 2 --gnn_aggregator pool \
  --seed 5 \
  --output_dir "outputs/sil_training_bi200_ss100_class1_legacy_gnn_pool_l2_hp_lr1e3_h64_d0_seed5" \
  > "sil_training_bi200_ss100_class1_legacy_gnn_pool_l2_hp_lr1e3_h64_d0_seed5.log" 2>&1
echo "=== [$(date +%H:%M:%S)] DONE: Pool 2L HP lr=0.001 h=64 do=0.0 seed5 (exit $?) ==="
