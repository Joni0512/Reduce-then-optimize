#!/bin/bash
# 2026-08-11: SIL (COAML) class-1-only (standard TRAINING_FILES/
# VALIDATION_FILES, no --extra_training_files/--extra_validation_files)
# Request Pruner sweep at bi400_ss200, thresholds 0.4/0.5/0.6 (this
# time), window-specific MLP checkpoint (see
# build_window_specific_request_pruners.py for why the Request Pruner,
# unlike the Pair Pruner, needs one model per window config). Baseline
# (no pruner) does not depend on threshold, so it is run once and reused
# for comparison against all three thresholds. --epochs 5
# --learning_rate 0.0001 set explicitly per thesis Table 3 (main.py
# defaults of 3/3e-4 do not match). Seed is an optional first CLI arg
# (default 42) so this can be repeated over multiple seeds.
set -u

SEED="${1:-42}"
REQUEST_PRUNER_MODEL="outputs/request_pruner_mlp_bi400_ss200/request_pruner_mlp_h32_l1_d0p0_pw1p0_lr0p03/request_pruner_mlp_h32_l1_d0p0_pw1p0_lr0p03_best_val_f3.pt"

echo "=== [$(date +%H:%M:%S)] SIL bi400_ss200 seed=${SEED} class1 requestpruner: baseline ==="
./venv/bin/python3 rtv_solver/main.py \
  --mode coaml \
  --input_dir "solutions/li_lim/manifests/" \
  --batch_interval 400 --step_size 200 --max_cardinality 2 \
  --learning_rate 0.0001 --epochs 5 \
  --use_request_pruner False --use_request_graph_pruner False \
  --seed "$SEED" \
  --output_dir "outputs/sil_training_bi400_ss200_class1_requestpruner_baseline_seed${SEED}" \
  > "sil_training_bi400_ss200_class1_requestpruner_baseline_seed${SEED}.log" 2>&1
echo "=== [$(date +%H:%M:%S)] DONE: baseline (exit $?) ==="

for THRESHOLD in 0.4 0.5 0.6; do
  echo "=== [$(date +%H:%M:%S)] SIL bi400_ss200 seed=${SEED} class1 requestpruner: request t=${THRESHOLD} ==="
  ./venv/bin/python3 rtv_solver/main.py \
    --mode coaml \
    --input_dir "solutions/li_lim/manifests/" \
    --batch_interval 400 --step_size 200 --max_cardinality 2 \
    --learning_rate 0.0001 --epochs 5 \
    --use_request_pruner True --request_pruner_model_path "$REQUEST_PRUNER_MODEL" --request_pruner_threshold "$THRESHOLD" --use_request_graph_pruner False \
    --seed "$SEED" \
    --output_dir "outputs/sil_training_bi400_ss200_class1_requestpruner_request_t${THRESHOLD}_seed${SEED}" \
    > "sil_training_bi400_ss200_class1_requestpruner_t${THRESHOLD}_seed${SEED}.log" 2>&1
  echo "=== [$(date +%H:%M:%S)] DONE: request t=${THRESHOLD} (exit $?) ==="
done

echo "=== ALL SIL bi400_ss200 seed=${SEED} class1 requestpruner RUNS DONE (4/4) ==="
