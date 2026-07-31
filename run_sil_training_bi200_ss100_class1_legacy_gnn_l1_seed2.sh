#!/bin/bash
# 2026-08-01: GNN seed-variant run (1 message-passing layer, seed 2) for the
# 1-vs-2-layer robustness comparison - same setup as
# run_sil_training_bi200_ss100_class1_legacy_gnn.sh (seed 42, 1 layer), only
# --seed and --gnn_num_message_passing_layers differ. Together with seed 42,
# this gives 5 seeds per layer count before trying the max-pooling aggregator.
set -eu

echo "=== [$(date +%H:%M:%S)] SIL bi200/ss100 class1 legacy - GNN 1L seed 2 ==="
./venv/bin/python3 rtv_solver/main.py \
  --mode coaml \
  --input_dir "solutions/li_lim/manifests/" \
  --batch_interval 200 --step_size 100 --max_cardinality 2 \
  --learning_rate 0.0001 --epochs 5 \
  --imitation_scoring_rule legacy \
  --use_request_pruner False --use_request_graph_pruner False \
  --model_type gnn --gnn_num_message_passing_layers 1 \
  --seed 2 \
  --output_dir "outputs/sil_training_bi200_ss100_class1_legacy_gnn_l1_seed2" \
  > "sil_training_bi200_ss100_class1_legacy_gnn_l1_seed2.log" 2>&1
echo "=== [$(date +%H:%M:%S)] DONE: class1 legacy GNN 1L seed 2 (exit $?) ==="
