#!/bin/bash
# 2026-08-11: SIL (COAML) MIXED-CLASS (both Li&Lim class 1 + class 2, via
# --extra_training_files/--extra_validation_files, lc204 excluded as a
# 160-200min/epoch outlier - see run_sil_training_bi200_ss100_mixedclass_t0.5.sh)
# Request Pruner sweep at bi200_ss100, thresholds 0.4/0.5/0.6, window-specific
# MLP checkpoint (see build_window_specific_request_pruners.py). Counterpart
# to run_sil_training_bi200_ss100_class1_requestpruner.sh (class-1-only version).
# Baseline (no pruner) run once per seed and reused across thresholds.
# --epochs 5 --learning_rate 0.0001 per thesis Table 3. Seed is an optional
# first CLI arg (default 42).
set -u

SEED="${1:-42}"
REQUEST_PRUNER_MODEL="outputs/request_pruner_mlp_bi200_ss100/request_pruner_mlp_h32_l1_d0p2_pw1p0_lr0p01/request_pruner_mlp_h32_l1_d0p2_pw1p0_lr0p01_best_val_f3.pt"
EXTRA_VAL="lc207,lc208,lr210,lr211,lrc207,lrc208"
EXTRA_TRAIN="lc201,lc202,lc203,lc205,lr201,lr202,lr203,lr204,lr205,lr206,lr207,lrc201,lrc202,lrc203,lrc204,lrc205"

echo "=== [$(date +%H:%M:%S)] SIL bi200_ss100 seed=${SEED} mixedclass requestpruner: baseline ==="
./venv/bin/python3 rtv_solver/main.py \
  --mode coaml \
  --input_dir "solutions/li_lim/manifests/" \
  --batch_interval 200 --step_size 100 --max_cardinality 2 \
  --learning_rate 0.0001 --epochs 5 \
  --extra_validation_files "$EXTRA_VAL" \
  --extra_training_files "$EXTRA_TRAIN" \
  --use_request_pruner False --use_request_graph_pruner False \
  --seed "$SEED" \
  --output_dir "outputs/sil_training_bi200_ss100_mixedclass_requestpruner_baseline_seed${SEED}" \
  > "sil_training_bi200_ss100_mixedclass_requestpruner_baseline_seed${SEED}.log" 2>&1
echo "=== [$(date +%H:%M:%S)] DONE: baseline (exit $?) ==="

for THRESHOLD in 0.4 0.5 0.6; do
  echo "=== [$(date +%H:%M:%S)] SIL bi200_ss100 seed=${SEED} mixedclass requestpruner: request t=${THRESHOLD} ==="
  ./venv/bin/python3 rtv_solver/main.py \
    --mode coaml \
    --input_dir "solutions/li_lim/manifests/" \
    --batch_interval 200 --step_size 100 --max_cardinality 2 \
    --learning_rate 0.0001 --epochs 5 \
    --extra_validation_files "$EXTRA_VAL" \
    --extra_training_files "$EXTRA_TRAIN" \
    --use_request_pruner True --request_pruner_model_path "$REQUEST_PRUNER_MODEL" --request_pruner_threshold "$THRESHOLD" --use_request_graph_pruner False \
    --seed "$SEED" \
    --output_dir "outputs/sil_training_bi200_ss100_mixedclass_requestpruner_request_t${THRESHOLD}_seed${SEED}" \
    > "sil_training_bi200_ss100_mixedclass_requestpruner_t${THRESHOLD}_seed${SEED}.log" 2>&1
  echo "=== [$(date +%H:%M:%S)] DONE: request t=${THRESHOLD} (exit $?) ==="
done

echo "=== ALL SIL bi200_ss100 seed=${SEED} mixedclass requestpruner RUNS DONE (4/4) ==="
