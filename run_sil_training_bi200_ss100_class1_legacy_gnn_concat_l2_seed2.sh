#!/bin/bash
# 2026-08-02: Ablation stage 2 (sum -> concat combine, GraphSAGE Algorithm 1)
# screening run, 2 message-passing layers, seed 2.
# Same setup as run_sil_training_bi200_ss100_class1_legacy_gnn_l2_seed2.sh,
# only --gnn_combine differs (concat instead of the sum-combine stage 0/1
# baseline). Runs alongside local seed 42/1 concat screens (this machine
# covers seeds 2/3) - together that is 4 of the eventual 6 seeds per
# 1L/2L-concat combination.
set -eu

echo "=== [$(date +%H:%M:%S)] SIL bi200/ss100 class1 legacy - GNN Concat 2L seed 2 ==="
./venv/bin/python3 rtv_solver/main.py \
  --mode coaml \
  --input_dir "solutions/li_lim/manifests/" \
  --batch_interval 200 --step_size 100 --max_cardinality 2 \
  --learning_rate 0.0001 --epochs 5 \
  --imitation_scoring_rule legacy \
  --use_request_pruner False --use_request_graph_pruner False \
  --model_type gnn --gnn_num_message_passing_layers 2 --gnn_combine concat \
  --seed 2 \
  --output_dir "outputs/sil_training_bi200_ss100_class1_legacy_gnn_concat_l2_seed2" \
  > "sil_training_bi200_ss100_class1_legacy_gnn_concat_l2_seed2.log" 2>&1
echo "=== [$(date +%H:%M:%S)] DONE: class1 legacy GNN Concat 2L seed 2 (exit $?) ==="
