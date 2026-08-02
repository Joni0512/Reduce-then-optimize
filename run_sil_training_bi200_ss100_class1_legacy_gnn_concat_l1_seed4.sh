#!/bin/bash
# 2026-08-02: Ablation stage 2 (sum -> concat combine) screening run, 1 message-
# passing layer, seed 4. Same setup as
# run_sil_training_bi200_ss100_class1_legacy_gnn_concat_l1_seed2/3.sh, submitted
# to the cm4 cluster (cm4_tiny) instead of serial - test run to check whether
# Gurobi allows this to run concurrently alongside the serial-cluster concat
# chain (submit_gnn_concat_seeds_chain.sh), or whether it hits the same
# concurrent-session license limit regardless of cluster.
set -eu

echo "=== [$(date +%H:%M:%S)] SIL bi200/ss100 class1 legacy - GNN Concat 1L seed4 (cm4) ==="
./venv/bin/python3 rtv_solver/main.py \
  --mode coaml \
  --input_dir "solutions/li_lim/manifests/" \
  --batch_interval 200 --step_size 100 --max_cardinality 2 \
  --learning_rate 0.0001 --epochs 5 \
  --imitation_scoring_rule legacy \
  --use_request_pruner False --use_request_graph_pruner False \
  --model_type gnn --gnn_num_message_passing_layers 1 --gnn_combine concat \
  --seed 4 \
  --output_dir "outputs/sil_training_bi200_ss100_class1_legacy_gnn_concat_l1_seed4" \
  > "sil_training_bi200_ss100_class1_legacy_gnn_concat_l1_seed4.log" 2>&1
echo "=== [$(date +%H:%M:%S)] DONE: class1 legacy GNN Concat 1L seed4 (exit $?) ==="
