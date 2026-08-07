#!/bin/bash
# 2026-08-07: Retry of pool-l2-sweep run wobbly-sweep-10 (0g4aqad7), which was
# cancelled by SLURM at the 6h time limit after only 2/5 epochs (h=256 is
# much slower per-epoch for Pool L2 than for GCN L1). Fixed HP combo run
# directly (not via wandb agent) with a 12h budget to get a complete result.
set -eu

echo "=== [$(date +%H:%M:%S)] SIL bi200/ss100 class1 legacy - GNN Pool 2L h256 retry4 (lr=0.003851657843037408) ==="
./venv/bin/python3 rtv_solver/main.py \
  --mode coaml \
  --input_dir "solutions/li_lim/manifests/" \
  --batch_interval 200 --step_size 100 --max_cardinality 2 \
  --learning_rate 0.003851657843037408 --epochs 5 \
  --hidden_dim 256 --dropout 0.0 \
  --imitation_scoring_rule legacy \
  --use_request_pruner False --use_request_graph_pruner False \
  --model_type gnn --gnn_num_message_passing_layers 2 --gnn_aggregator pool \
  --seed 1 \
  --output_dir "outputs/sil_training_bi200_ss100_class1_legacy_gnn_pool_l2_h256_retry4" \
  > "sil_training_bi200_ss100_class1_legacy_gnn_pool_l2_h256_retry4.log" 2>&1
echo "=== [$(date +%H:%M:%S)] DONE: Pool 2L h256 retry4 (exit $?) ==="
