#!/bin/bash
# 2026-08-05: same setup as run_sil_training_bi200_ss100_pairpruner_clean_t0.5.sh
# (Pair Pruner v1_final GNN, bi200/ss100, t=0.5, mixed-class train/val split,
# 5 epochs, lr=0.0001) but with --model_type gnn instead of the default
# ScoringMLP, i.e. the RTV-candidate scoring network for the SIL assignment
# ILP is now CandidateScoringGNN (rtv_solver/pipeline/candidate_scoring_gnn.py)
# instead of the simple item-wise MLP. Everything else identical, so results
# are directly comparable against the MLP-scoring run.
set -u

PAIR_PRUNER_MODEL="outputs/models_v2/rgnn_mixed_c2_pw2_v1_final/rgnn_mixed_c2_pw2_v1_final_best_val_f3.pt"
THRESHOLD=0.5
EXTRA_VAL="lc207,lc208,lr210,lr211,lrc207,lrc208"
EXTRA_TRAIN="lc201,lc202,lc203,lc205,lr201,lr202,lr203,lr204,lr205,lr206,lr207,lrc201,lrc202,lrc203,lrc204,lrc205"

echo "=== [$(date +%H:%M:%S)] SIL bi200/ss100 pairpruner-clean-gnnscore t=0.5: baseline ==="
./venv/bin/python3 rtv_solver/main.py \
  --mode coaml \
  --input_dir "solutions/li_lim/manifests/" \
  --batch_interval 200 --step_size 100 --max_cardinality 2 \
  --learning_rate 0.0001 --epochs 5 \
  --model_type gnn \
  --extra_validation_files "$EXTRA_VAL" \
  --extra_training_files "$EXTRA_TRAIN" \
  --use_request_pruner False --use_request_graph_pruner False \
  --seed 42 \
  --output_dir "outputs/sil_training_bi200_ss100_pairpruner_clean_gnnscore_baseline_t0.5" \
  > "sil_training_bi200_ss100_pairpruner_clean_gnnscore_baseline_t0.5.log" 2>&1
echo "=== [$(date +%H:%M:%S)] DONE: baseline (exit $?) ==="

echo "=== [$(date +%H:%M:%S)] SIL bi200/ss100 pairpruner-clean-gnnscore t=0.5: pair ==="
./venv/bin/python3 rtv_solver/main.py \
  --mode coaml \
  --input_dir "solutions/li_lim/manifests/" \
  --batch_interval 200 --step_size 100 --max_cardinality 2 \
  --learning_rate 0.0001 --epochs 5 \
  --model_type gnn \
  --extra_validation_files "$EXTRA_VAL" \
  --extra_training_files "$EXTRA_TRAIN" \
  --use_request_pruner False --use_request_graph_pruner True --request_graph_model_path "$PAIR_PRUNER_MODEL" --request_graph_threshold "$THRESHOLD" \
  --seed 42 \
  --output_dir "outputs/sil_training_bi200_ss100_pairpruner_clean_gnnscore_pair_t0.5" \
  > "sil_training_bi200_ss100_pairpruner_clean_gnnscore_pair_t0.5.log" 2>&1
echo "=== [$(date +%H:%M:%S)] DONE: pair (exit $?) ==="

echo "=== ALL SIL bi200/ss100 pairpruner-clean-gnnscore t=0.5 RUNS DONE ==="
