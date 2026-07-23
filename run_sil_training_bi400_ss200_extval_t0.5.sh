#!/bin/bash
# 2026-07-19: SIL sweep at bi400/ss200, t=0.5, but with VALIDATION_FILES
# extended to include the LC2/LR2/LRC2 test instances (lc207,lc208,lr210,
# lr211,lrc207,lrc208) via the new --extra_validation_files flag. These 6
# instances are part of the pruners' own test set (MODEL_CONFIGS["mixed_all"])
# but were never covered by the standard 6-file SIL VALIDATION_FILES - every
# SIL result so far in this session only used LC1/LR1/LRC1. This checks
# whether the damage-mechanism findings hold on the other half of the
# classifier's held-out test set.
# TRAINING_FILES is untouched - this does not retrain differently, only
# evaluates on 6 more instances per epoch (12 total instead of 6).
set -u

REQUEST_PRUNER_MODEL="outputs/request_pruner_mlp_bi400_ss200/request_pruner_mlp_h32_l1_d0p0_pw1p0_lr0p03/request_pruner_mlp_h32_l1_d0p0_pw1p0_lr0p03_best_val_f3.pt"
THRESHOLD=0.5
EXTRA_VAL="lc207,lc208,lr210,lr211,lrc207,lrc208"

run_variant() {
  local variant="$1" pruner_flags="$2"
  echo "=== [$(date +%H:%M:%S)] SIL bi400/ss200 extval t=0.5: $variant ==="
  ./venv/bin/python3 rtv_solver/main.py \
    --mode coaml \
    --input_dir "solutions/li_lim/manifests/" \
    --batch_interval 400 --step_size 200 --max_cardinality 2 \
    --learning_rate 0.0001 --epochs 5 \
    --extra_validation_files "$EXTRA_VAL" \
    $pruner_flags \
    --seed 42 \
    --output_dir "outputs/sil_training_bi400_ss200_extval_${variant}_t0.5" \
    > "sil_training_bi400_ss200_extval_${variant}_t0.5.log" 2>&1
  echo "=== [$(date +%H:%M:%S)] DONE: $variant (exit $?) ==="
}

run_variant "baseline" "--use_request_pruner False --use_request_graph_pruner False"
run_variant "request" "--use_request_pruner True --request_pruner_model_path $REQUEST_PRUNER_MODEL --request_pruner_threshold $THRESHOLD --use_request_graph_pruner False"
run_variant "pair" "--use_request_pruner False --use_request_graph_pruner True --request_graph_threshold $THRESHOLD"
run_variant "both" "--use_request_pruner True --request_pruner_model_path $REQUEST_PRUNER_MODEL --request_pruner_threshold $THRESHOLD --use_request_graph_pruner True --request_graph_threshold $THRESHOLD"

echo "=== ALL 4 SIL bi400/ss200 extval t=0.5 RUNS DONE ==="
