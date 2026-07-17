#!/bin/bash
# Baseline-only (pruner off) for RH + COAML, all 12 instances, for a given
# window config. Separate from run_final_compare_window.sh so it can be run
# standalone as a quick timing probe before committing to the full v1/v2 sweep.
#
# Usage: ./run_baseline_only_window.sh <batch_interval> <step_size> [seed]
# Example: ./run_baseline_only_window.sh 200 100        (seed defaults to 42)
# Example: ./run_baseline_only_window.sh 200 100 1       (seed 1 -> folder suffix _seed1)
set -u

BATCH_INTERVAL="${1:?usage: run_baseline_only_window.sh <batch_interval> <step_size> [seed]}"
STEP_SIZE="${2:?usage: run_baseline_only_window.sh <batch_interval> <step_size> [seed]}"
SEED="${3:-42}"
SEED_SUFFIX=""
[ "$SEED" != "42" ] && SEED_SUFFIX="_seed${SEED}"

BASE_OUT="new_tests/final_compare/bi${BATCH_INTERVAL}_ss${STEP_SIZE}"
INSTANCES="lc108 lc109 lc207 lc208 lr111 lr112 lr210 lr211 lrc107 lrc108 lrc207 lrc208"
NOPRUNE_MODEL="outputs/models_v2_gnnv2/rgnn_mixed_c2_pw10_v2/rgnn_mixed_c2_pw10_v2_best_val_f3.pt"

echo "=== RH baseline (12 runs), bi${BATCH_INTERVAL}_ss${STEP_SIZE}, seed=${SEED} ==="
for inst in $INSTANCES; do
  echo ">>> rh/baseline${SEED_SUFFIX}/${inst}"
  time python rtv_solver/main.py \
    --mode offline \
    --input_file "solutions/li_lim/manifests/${inst}.json" \
    --input_dir "solutions/li_lim/manifests/" \
    --imitation_solution_file "solutions/li_lim/manifests/${inst}.json" \
    --use_request_graph_pruner False \
    --request_graph_model_path "$NOPRUNE_MODEL" \
    --request_graph_threshold 0.5 \
    --seed "$SEED" \
    --max_cardinality 2 --batch_interval "$BATCH_INTERVAL" --step_size "$STEP_SIZE" \
    --output_dir "${BASE_OUT}/rh_baseline${SEED_SUFFIX}/${inst}"
done

echo "=== COAML baseline (12 runs), bi${BATCH_INTERVAL}_ss${STEP_SIZE}, seed=${SEED} ==="
for inst in $INSTANCES; do
  echo ">>> coaml/baseline${SEED_SUFFIX}/${inst}"
  time python rtv_solver/main.py \
    --mode coaml --input_dir "" \
    --input_file "solutions/li_lim/manifests/${inst}.json" \
    --imitation_solution_file "solutions/li_lim/manifests/${inst}.json" \
    --use_request_graph_pruner False \
    --request_graph_model_path "$NOPRUNE_MODEL" \
    --request_graph_threshold 0.5 \
    --seed "$SEED" \
    --max_cardinality 2 --batch_interval "$BATCH_INTERVAL" --step_size "$STEP_SIZE" \
    --output_dir "${BASE_OUT}/coaml_baseline${SEED_SUFFIX}/${inst}"
done

echo "=== BASELINE DONE (24/24), bi${BATCH_INTERVAL}_ss${STEP_SIZE}, seed=${SEED} ==="
