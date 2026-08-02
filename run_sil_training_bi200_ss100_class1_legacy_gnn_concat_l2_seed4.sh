#!/bin/bash
# 2026-08-02: Ablation stage 2 (sum -> concat combine) screening run, 2 message-
# passing layers, seed 4. Parallel-execution production run on serial_long
# (already validated to work concurrently alongside serial_std/cm4_tiny jobs) -
# 2-layer is behind 1-layer in seed count, this closes part of the gap toward
# 5 seeds per layer count today.
set -eu

echo "=== [$(date +%H:%M:%S)] SIL bi200/ss100 class1 legacy - GNN Concat 2L seed4 (serial_long) ==="
./venv/bin/python3 rtv_solver/main.py \
  --mode coaml \
  --input_dir "solutions/li_lim/manifests/" \
  --batch_interval 200 --step_size 100 --max_cardinality 2 \
  --learning_rate 0.0001 --epochs 5 \
  --imitation_scoring_rule legacy \
  --use_request_pruner False --use_request_graph_pruner False \
  --model_type gnn --gnn_num_message_passing_layers 2 --gnn_combine concat \
  --seed 4 \
  --output_dir "outputs/sil_training_bi200_ss100_class1_legacy_gnn_concat_l2_seed4" \
  > "sil_training_bi200_ss100_class1_legacy_gnn_concat_l2_seed4.log" 2>&1
echo "=== [$(date +%H:%M:%S)] DONE: class1 legacy GNN Concat 2L seed4 (exit $?) ==="
