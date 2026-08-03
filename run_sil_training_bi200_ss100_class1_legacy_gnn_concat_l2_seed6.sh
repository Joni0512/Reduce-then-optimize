#!/bin/bash
# 2026-08-03: Ablation stage 2 (concat combine, GraphSAGE Algorithm 1)
# screening run, 2 message-passing layers, seed 6. Extends the L2 seed set
# (2,3,4,5) to 5 seeds total (2,3,4,5,6), matching the constant-5-seeds
# target across all layer/aggregator combinations.
set -eu

echo "=== [$(date +%H:%M:%S)] SIL bi200/ss100 class1 legacy - GNN Concat 2L seed 6 ==="
./venv/bin/python3 rtv_solver/main.py \
  --mode coaml \
  --input_dir "solutions/li_lim/manifests/" \
  --batch_interval 200 --step_size 100 --max_cardinality 2 \
  --learning_rate 0.0001 --epochs 5 \
  --imitation_scoring_rule legacy \
  --use_request_pruner False --use_request_graph_pruner False \
  --model_type gnn --gnn_num_message_passing_layers 2 --gnn_aggregator mean \
  --seed 6 \
  --output_dir "outputs/sil_training_bi200_ss100_class1_legacy_gnn_concat_l2_seed6" \
  > "sil_training_bi200_ss100_class1_legacy_gnn_concat_l2_seed6.log" 2>&1
echo "=== [$(date +%H:%M:%S)] DONE: class1 legacy GNN Concat 2L seed 6 (exit $?) ==="
