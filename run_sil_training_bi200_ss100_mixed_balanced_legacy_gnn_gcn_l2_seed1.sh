#!/bin/bash
# 2026-09-07: GNN-actor variant of
# run_sil_training_bi200_ss100_mixed_balanced_legacy_mlp_seed1.sh - see chat.
# Same balanced mixed (class1+class2, lc/lr/lrc) train/val split, same
# hyperparameters (learning_rate, epochs, batch_interval, step_size,
# cardinality, scoring rule) - only model_type switches from mlp to gnn
# (gcn aggregator, 2 message-passing layers, matching one of the class1-only
# GNN screening runs). Motivation: actor-GNN outperformed actor-MLP in
# earlier class1-only SIL comparisons; testing whether that holds on the
# mixed balanced split too, as a prerequisite before using an actor-GNN
# checkpoint for SRL/GAT fine-tuning. Hyperparameters left at these
# defaults for now (not yet swept) - sweep later if this direction looks
# promising.
set -eu

echo "=== [$(date +%H:%M:%S)] SIL bi200/ss100 mixed BALANCED (class1+class2, lc/lr/lrc) legacy GNN gcn-l2 seed 1 ==="
./venv/bin/python3 rtv_solver/main.py \
  --mode coaml \
  --input_dir "solutions/li_lim/manifests/" \
  --override_training_files "lc101,lc102,lr101,lr102,lrc101,lrc102,lc201,lc202,lr201,lr202,lrc202,lrc203" \
  --override_validation_files "lc107,lc108,lr111,lr112,lrc107,lrc108,lc207,lc208,lr203,lr210,lrc201,lrc207" \
  --batch_interval 200 --step_size 100 --max_cardinality 2 \
  --learning_rate 0.0001 --epochs 5 \
  --imitation_scoring_rule legacy \
  --use_request_pruner False --use_request_graph_pruner False \
  --model_type gnn --gnn_num_message_passing_layers 2 --gnn_aggregator gcn \
  --seed 1 \
  --output_dir "outputs/sil_training_bi200_ss100_mixed_balanced_legacy_gnn_gcn_l2_seed1" \
  > "sil_training_bi200_ss100_mixed_balanced_legacy_gnn_gcn_l2_seed1.log" 2>&1
echo "=== [$(date +%H:%M:%S)] DONE: mixed balanced legacy GNN gcn-l2 seed 1 (exit $?) ==="
