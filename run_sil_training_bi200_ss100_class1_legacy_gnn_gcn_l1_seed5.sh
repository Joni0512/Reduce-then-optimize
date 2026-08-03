#!/bin/bash
# 2026-08-02: GCN aggregator screening run (Eq. 2, self+neighbours meaned
# together before one shared linear layer), 1 message-passing layer(s), seed 5.
# Replaces the old (pre-fix) sum-combine seed-5/L1 run: ConvolutionalMeanLayer
# used two separate linears then summed, which was not the true GCN-style
# aggregator - GCNMeanLayer corrects this. Seed set aligned with the
# GraphSAGE-mean side (concat_l1_seed2-5) for a paired comparison.
set -eu

echo "=== [$(date +%H:%M:%S)] SIL bi200/ss100 class1 legacy - GNN GCN 1L seed 5 ==="
./venv/bin/python3 rtv_solver/main.py \
  --mode coaml \
  --input_dir "solutions/li_lim/manifests/" \
  --batch_interval 200 --step_size 100 --max_cardinality 2 \
  --learning_rate 0.0001 --epochs 5 \
  --imitation_scoring_rule legacy \
  --use_request_pruner False --use_request_graph_pruner False \
  --model_type gnn --gnn_num_message_passing_layers 1 --gnn_aggregator gcn \
  --seed 5 \
  --output_dir "outputs/sil_training_bi200_ss100_class1_legacy_gnn_gcn_l1_seed5" \
  > "sil_training_bi200_ss100_class1_legacy_gnn_gcn_l1_seed5.log" 2>&1
echo "=== [$(date +%H:%M:%S)] DONE: class1 legacy GNN GCN 1L seed 5 (exit $?) ==="
