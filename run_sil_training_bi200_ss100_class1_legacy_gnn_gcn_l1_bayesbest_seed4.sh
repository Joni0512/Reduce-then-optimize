#!/bin/bash
# 2026-08-05: GCN L1 confirmation run - best combo from wandb Bayes sweep
# (gcn-l1-sweep/np8vurt5, run xgxwhk8o: lr=0.0006429275369146481, h=256,
# dropout=0.0, 78.30% val service rate at seed=1). Confirming across 4 more
# seeds (2-5) to check the result holds beyond a single seed.
set -eu

echo "=== [$(date +%H:%M:%S)] SIL bi200/ss100 class1 legacy - GNN GCN 1L Bayes-best seed 4 ==="
./venv/bin/python3 rtv_solver/main.py \
  --mode coaml \
  --input_dir "solutions/li_lim/manifests/" \
  --batch_interval 200 --step_size 100 --max_cardinality 2 \
  --learning_rate 0.0006429275369146481 --epochs 5 \
  --hidden_dim 256 --dropout 0.0 \
  --imitation_scoring_rule legacy \
  --use_request_pruner False --use_request_graph_pruner False \
  --model_type gnn --gnn_num_message_passing_layers 1 --gnn_aggregator gcn \
  --seed 4 \
  --output_dir "outputs/sil_training_bi200_ss100_class1_legacy_gnn_gcn_l1_bayesbest_seed4" \
  > "sil_training_bi200_ss100_class1_legacy_gnn_gcn_l1_bayesbest_seed4.log" 2>&1
echo "=== [$(date +%H:%M:%S)] DONE: GCN 1L Bayes-best seed 4 (exit $?) ==="
