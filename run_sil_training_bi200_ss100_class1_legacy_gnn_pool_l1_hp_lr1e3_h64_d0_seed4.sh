#!/bin/bash
# 2026-08-04: HP-tuning stage 2 confirmation run - Pool L1, lr=0.001, h=64,
# dropout=0.0, seed 4. Testing whether the higher-learning-rate trend
# found for GCN L1 (0.0001 -> 0.001 was a large, consistent improvement)
# also holds for Pool - seed1 for this combo is already running on the
# cluster as part of the 9-combo grid.
set -eu

echo "=== [$(date +%H:%M:%S)] SIL bi200/ss100 class1 legacy - GNN Pool 1L HP lr=0.001 h=64 do=0.0 seed4 ==="
./venv/bin/python3 rtv_solver/main.py \
  --mode coaml \
  --input_dir "solutions/li_lim/manifests/" \
  --batch_interval 200 --step_size 100 --max_cardinality 2 \
  --learning_rate 0.001 --epochs 5 \
  --hidden_dim 64 --dropout 0.0 \
  --imitation_scoring_rule legacy \
  --use_request_pruner False --use_request_graph_pruner False \
  --model_type gnn --gnn_num_message_passing_layers 1 --gnn_aggregator pool \
  --seed 4 \
  --output_dir "outputs/sil_training_bi200_ss100_class1_legacy_gnn_pool_l1_hp_lr1e3_h64_d0_seed4" \
  > "sil_training_bi200_ss100_class1_legacy_gnn_pool_l1_hp_lr1e3_h64_d0_seed4.log" 2>&1
echo "=== [$(date +%H:%M:%S)] DONE: Pool 1L HP lr=0.001 h=64 do=0.0 seed4 (exit $?) ==="
