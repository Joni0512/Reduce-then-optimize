#!/bin/bash
# 2026-08-04: HP-tuning stage 2 confirmation run - GCN L1, lr=0.001, h=128,
# dropout=0.0, seed 3. Best single-seed combo from the coarse HP screen
# (seed1: 77.99%, far above the Table-3-HP baseline of ~73%) - running 4 more
# seeds (2-5) to confirm the trend holds and isn't just seed1 noise.
set -eu

echo "=== [$(date +%H:%M:%S)] SIL bi200/ss100 class1 legacy - GNN GCN 1L HP lr=0.001 h=128 do=0.0 seed3 ==="
./venv/bin/python3 rtv_solver/main.py \
  --mode coaml \
  --input_dir "solutions/li_lim/manifests/" \
  --batch_interval 200 --step_size 100 --max_cardinality 2 \
  --learning_rate 0.001 --epochs 5 \
  --hidden_dim 128 --dropout 0.0 \
  --imitation_scoring_rule legacy \
  --use_request_pruner False --use_request_graph_pruner False \
  --model_type gnn --gnn_num_message_passing_layers 1 --gnn_aggregator gcn \
  --seed 3 \
  --output_dir "outputs/sil_training_bi200_ss100_class1_legacy_gnn_gcn_l1_hp_lr1e3_h128_d0_seed3" \
  > "sil_training_bi200_ss100_class1_legacy_gnn_gcn_l1_hp_lr1e3_h128_d0_seed3.log" 2>&1
echo "=== [$(date +%H:%M:%S)] DONE: GCN 1L HP lr=0.001 h=128 do=0.0 seed3 (exit $?) ==="
