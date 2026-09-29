#!/bin/bash
# GNN counterpart to run_sil_training_bi200_ss100_clean_srl_split_legacy_mlp_seed1.sh - same
# clean (no VAL/TEST leakage) 32/6 split, but --model_type gnn with the best confirmed GNN
# config from the earlier architecture screening (see chat): GCN, 1 message-passing layer,
# hidden_dim=128, lr=0.001, feature_builder v1 with cr_pickup_slack enabled (5-seed mean
# 78.93% +/- 0.32% on that earlier, DIFFERENT split - this run uses the clean SRL split
# instead, so the resulting number is not directly comparable to that 78.93%).
set -eu

echo "=== [$(date +%H:%M:%S)] SIL bi200/ss100 clean SRL-split (no VAL/TEST leakage) legacy GNN GCN L1 seed 1 ==="
./venv/bin/python3 rtv_solver/main.py \
  --mode coaml \
  --input_dir "solutions/li_lim/manifests/" \
  --override_training_files "lc101,lc102,lc105,lc106,lc107,lc109,lc201,lc203,lc206,lc207,lr101,lr102,lr105,lr106,lr108,lr109,lr110,lr112,lr201,lr203,lr206,lr207,lr209,lr210,lrc101,lrc103,lrc106,lrc107,lrc201,lrc203,lrc206,lrc207" \
  --override_validation_files "lc104,lc204,lr104,lr204,lrc104,lrc204" \
  --batch_interval 200 --step_size 100 --max_cardinality 2 \
  --learning_rate 0.001 --epochs 5 \
  --imitation_scoring_rule legacy \
  --use_request_pruner False --use_request_graph_pruner False \
  --model_type gnn --gnn_aggregator gcn --gnn_num_message_passing_layers 1 --hidden_dim 128 \
  --feature_builder_version v1 --enable_pickup_slack_feature True \
  --seed 1 \
  --output_dir "outputs/sil_training_bi200_ss100_clean_srl_split_legacy_gnn_gcn_l1_seed1" \
  > "sil_training_bi200_ss100_clean_srl_split_legacy_gnn_gcn_l1_seed1.log" 2>&1
echo "=== [$(date +%H:%M:%S)] DONE: clean SRL-split legacy GNN GCN L1 seed 1 (exit $?) ==="
