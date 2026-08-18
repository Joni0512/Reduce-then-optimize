#!/bin/bash
# 2026-08-18: RHO WITHOUT learning (--mode offline, myopic rolling-horizon,
# no SIL/ML training), Request Pruner at bi200/ss100, FIXED threshold=0.5,
# sweeping min_retention_fraction (the retention-floor safety measure - see
# request_pruner.py's min_retention_fraction / --request_pruner_min_retention_fraction,
# newly exposed via CLI) over 0.3/0.4/0.5/0.6. Class-1-only 6-instance set,
# same window-specific MLP checkpoint as the other bi200/ss100 request-pruner
# sweeps. Seed is an optional first CLI arg (default 42).
set -u

BATCH_INTERVAL=200
STEP_SIZE=100
SEED="${1:-42}"
THRESHOLD=0.5
REQUEST_PRUNER_MODEL="outputs/request_pruner_mlp_bi200_ss100/request_pruner_mlp_h32_l1_d0p2_pw1p0_lr0p01/request_pruner_mlp_h32_l1_d0p2_pw1p0_lr0p01_best_val_f3.pt"
INSTANCES="lc108 lc109 lr111 lr112 lrc107 lrc108"
BASE_OUT="new_tests/rh_bi200_ss100_class1_requestpruner_retentionsweep_seed${SEED}"

for MIN_RETENTION in 0.3 0.4 0.5 0.6; do
  echo "=== [$(date +%H:%M:%S)] RH bi200_ss100 seed=${SEED} requestpruner t=${THRESHOLD} min_retention=${MIN_RETENTION} ==="
  for inst in $INSTANCES; do
    echo ">>> minret${MIN_RETENTION}/${inst}"
    ./venv/bin/python3 rtv_solver/main.py \
      --mode offline \
      --input_file "solutions/li_lim/manifests/${inst}.json" \
      --input_dir "solutions/li_lim/manifests/" \
      --imitation_solution_file "solutions/li_lim/manifests/${inst}.json" \
      --use_request_pruner True --request_pruner_model_path "$REQUEST_PRUNER_MODEL" --request_pruner_threshold "$THRESHOLD" \
      --request_pruner_min_retention_fraction "$MIN_RETENTION" \
      --seed "$SEED" \
      --max_cardinality 2 --batch_interval "$BATCH_INTERVAL" --step_size "$STEP_SIZE" \
      --output_dir "${BASE_OUT}/minret${MIN_RETENTION}/${inst}"
  done
  echo "=== [$(date +%H:%M:%S)] DONE: min_retention=${MIN_RETENTION} ==="
done

echo "=== ALL RH bi200_ss100 seed=${SEED} requestpruner retention-sweep RUNS DONE (4 variants x 6 instances) ==="
