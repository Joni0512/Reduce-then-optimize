#!/bin/bash
# 2026-08-16: RHO WITHOUT learning (--mode offline, myopic rolling-horizon
# heuristic, no SIL/ML training), CLASS-2-ONLY (6 Li&Lim class-2 val
# instances: lc207,lc208,lr210,lr211,lrc207,lrc208). Counterpart to
# run_rh_bi400_ss200_class1_requestpruner.sh (class-1-only version) and
# run_rh_bi400_ss200_class2only_pairpruner_v1final.sh (class-2-only Pair Pruner
# version). Window-specific Request Pruner MLP checkpoint (see
# build_window_specific_request_pruners.py). Thresholds 0.4/0.5/0.6,
# baseline run once per instance and reused across thresholds. Seed is an
# optional first CLI arg (default 42).
set -u

BATCH_INTERVAL=400
STEP_SIZE=200
SEED="${1:-42}"
REQUEST_PRUNER_MODEL="outputs/request_pruner_mlp_bi400_ss200/request_pruner_mlp_h32_l1_d0p0_pw1p0_lr0p03/request_pruner_mlp_h32_l1_d0p0_pw1p0_lr0p03_best_val_f3.pt"
INSTANCES="lc207 lc208 lr210 lr211 lrc207 lrc208"
BASE_OUT="new_tests/rh_class2only_requestpruner/bi400_ss200_seed${SEED}"

echo "=== [$(date +%H:%M:%S)] RH bi400_ss200 seed=${SEED} class2only requestpruner: baseline ==="
for inst in $INSTANCES; do
  echo ">>> baseline/${inst}"
  ./venv/bin/python3 rtv_solver/main.py \
    --mode offline \
    --input_file "solutions/li_lim/manifests/${inst}.json" \
    --input_dir "solutions/li_lim/manifests/" \
    --imitation_solution_file "solutions/li_lim/manifests/${inst}.json" \
    --use_request_pruner False \
    --seed "$SEED" \
    --max_cardinality 2 --batch_interval "$BATCH_INTERVAL" --step_size "$STEP_SIZE" \
    --output_dir "${BASE_OUT}/baseline/${inst}"
done
echo "=== [$(date +%H:%M:%S)] DONE: baseline ==="

for THRESHOLD in 0.4 0.5 0.6; do
  echo "=== [$(date +%H:%M:%S)] RH bi400_ss200 seed=${SEED} class2only requestpruner: request t=${THRESHOLD} ==="
  for inst in $INSTANCES; do
    echo ">>> request_t${THRESHOLD}/${inst}"
    ./venv/bin/python3 rtv_solver/main.py \
      --mode offline \
      --input_file "solutions/li_lim/manifests/${inst}.json" \
      --input_dir "solutions/li_lim/manifests/" \
      --imitation_solution_file "solutions/li_lim/manifests/${inst}.json" \
      --use_request_pruner True --request_pruner_model_path "$REQUEST_PRUNER_MODEL" --request_pruner_threshold "$THRESHOLD" \
      --seed "$SEED" \
      --max_cardinality 2 --batch_interval "$BATCH_INTERVAL" --step_size "$STEP_SIZE" \
      --output_dir "${BASE_OUT}/request_t${THRESHOLD}/${inst}"
  done
  echo "=== [$(date +%H:%M:%S)] DONE: request t=${THRESHOLD} ==="
done

echo "=== ALL RH bi400_ss200 seed=${SEED} class2only requestpruner RUNS DONE (4 variants x 6 instances) ==="
