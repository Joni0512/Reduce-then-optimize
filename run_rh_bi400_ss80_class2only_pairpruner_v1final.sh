#!/bin/bash
# 2026-08-16: RHO WITHOUT learning (--mode offline, myopic rolling-horizon
# heuristic, no SIL/ML training), CLASS-2-ONLY (6 Li&Lim class-2 val
# instances: lc207,lc208,lr210,lr211,lrc207,lrc208 - same 6 used as
# EXTRA_VALIDATION_FILES in the mixed-class SIL runs). Counterpart to
# run_rh_bi400_ss80_class1_pairpruner_v1final.sh (class-1-only version). Pair
# Pruner (GNN v1_final), thresholds 0.4/0.5/0.6. Baseline (no pruner) run
# once per instance and reused for comparison against all three thresholds.
# Seed is an optional first CLI arg (default 42).
set -u

BATCH_INTERVAL=400
STEP_SIZE=80
SEED="${1:-42}"
PAIR_PRUNER_MODEL="outputs/models_v2/rgnn_mixed_c2_pw2_v1_final/rgnn_mixed_c2_pw2_v1_final_best_val_f3.pt"
INSTANCES="lc207 lc208 lr210 lr211 lrc207 lrc208"
BASE_OUT="new_tests/rh_class2only_pairpruner_v1final/bi400_ss80_seed${SEED}"

echo "=== [$(date +%H:%M:%S)] RH bi400_ss80 seed=${SEED} class2only pairpruner v1final: baseline ==="
for inst in $INSTANCES; do
  echo ">>> baseline/${inst}"
  ./venv/bin/python3 rtv_solver/main.py \
    --mode offline \
    --input_file "solutions/li_lim/manifests/${inst}.json" \
    --input_dir "solutions/li_lim/manifests/" \
    --imitation_solution_file "solutions/li_lim/manifests/${inst}.json" \
    --use_request_graph_pruner False \
    --seed "$SEED" \
    --max_cardinality 2 --batch_interval "$BATCH_INTERVAL" --step_size "$STEP_SIZE" \
    --output_dir "${BASE_OUT}/baseline/${inst}"
done
echo "=== [$(date +%H:%M:%S)] DONE: baseline ==="

for THRESHOLD in 0.4 0.5 0.6; do
  echo "=== [$(date +%H:%M:%S)] RH bi400_ss80 seed=${SEED} class2only pairpruner v1final: pair t=${THRESHOLD} ==="
  for inst in $INSTANCES; do
    echo ">>> pair_t${THRESHOLD}/${inst}"
    ./venv/bin/python3 rtv_solver/main.py \
      --mode offline \
      --input_file "solutions/li_lim/manifests/${inst}.json" \
      --input_dir "solutions/li_lim/manifests/" \
      --imitation_solution_file "solutions/li_lim/manifests/${inst}.json" \
      --use_request_graph_pruner True --request_graph_model_path "$PAIR_PRUNER_MODEL" --request_graph_threshold "$THRESHOLD" \
      --seed "$SEED" \
      --max_cardinality 2 --batch_interval "$BATCH_INTERVAL" --step_size "$STEP_SIZE" \
      --output_dir "${BASE_OUT}/pair_t${THRESHOLD}/${inst}"
  done
  echo "=== [$(date +%H:%M:%S)] DONE: pair t=${THRESHOLD} ==="
done

echo "=== ALL RH bi400_ss80 seed=${SEED} class2only pairpruner v1final RUNS DONE (4 variants x 6 instances) ==="
