#!/bin/bash
# Generalized version of run_final_compare_t04.sh: RH/COAML v1_final vs
# v2_final comparison at an arbitrary threshold, same seed (42) throughout.
# Baseline (pruner off) is threshold-independent, so it is NOT rerun --
# reuses outputs/new_tests/final_compare/bi{BI}_ss{SS}/{rh,coaml}_baseline
# from the threshold-0.5 sweep. Saves cleanly under a single outputs/.
#
# Usage: ./run_final_compare_threshold.sh 0.3
set -u

THRESHOLD="${1:?usage: run_final_compare_threshold.sh <threshold e.g. 0.3>}"
TSUFFIX="${THRESHOLD/./}"   # 0.3 -> 03, 0.25 -> 025

BATCH_INTERVAL=400
STEP_SIZE=100
BASE_OUT="new_tests/final_compare/bi${BATCH_INTERVAL}_ss${STEP_SIZE}"

INSTANCES="lc108 lc109 lc207 lc208 lr111 lr112 lr210 lr211 lrc107 lrc108 lrc207 lrc208"

V1_MODEL="outputs/models_v2/rgnn_mixed_c2_pw2_v1_final/rgnn_mixed_c2_pw2_v1_final_best_val_f3.pt"
V2_MODEL="outputs/models_v2_gnnv2/rgnn_mixed_c2_pw2_v2_final/rgnn_mixed_c2_pw2_v2_final_best_val_f3.pt"

FAILED=()

run_rh() {
  local method="$1" model="$2" inst="$3"
  local out_dir="${BASE_OUT}/rh_${method}_t${TSUFFIX}/${inst}"
  echo ">>> rh/${method}_t${TSUFFIX}/${inst}"
  python rtv_solver/main.py \
    --mode offline \
    --input_file "solutions/li_lim/manifests/${inst}.json" \
    --input_dir "solutions/li_lim/manifests/" \
    --imitation_solution_file "solutions/li_lim/manifests/${inst}.json" \
    --use_request_graph_pruner True \
    --request_graph_model_path "$model" \
    --request_graph_threshold "$THRESHOLD" \
    --max_cardinality 2 --batch_interval "$BATCH_INTERVAL" --step_size "$STEP_SIZE" \
    --output_dir "$out_dir" \
    > "/tmp/rh_${method}_t${TSUFFIX}_${inst}.log" 2>&1
  if [ $? -ne 0 ]; then
    echo "FAILED: rh/${method}_t${TSUFFIX}/${inst} (see /tmp/rh_${method}_t${TSUFFIX}_${inst}.log)"
    FAILED+=("rh/${method}_t${TSUFFIX}/${inst}")
  fi
}

run_coaml() {
  local method="$1" model="$2" inst="$3"
  local out_dir="${BASE_OUT}/coaml_${method}_t${TSUFFIX}/${inst}"
  echo ">>> coaml/${method}_t${TSUFFIX}/${inst}"
  python rtv_solver/main.py \
    --mode coaml --input_dir "" \
    --input_file "solutions/li_lim/manifests/${inst}.json" \
    --imitation_solution_file "solutions/li_lim/manifests/${inst}.json" \
    --use_request_graph_pruner True \
    --request_graph_model_path "$model" \
    --request_graph_threshold "$THRESHOLD" \
    --max_cardinality 2 --batch_interval "$BATCH_INTERVAL" --step_size "$STEP_SIZE" \
    --output_dir "$out_dir" \
    > "/tmp/coaml_${method}_t${TSUFFIX}_${inst}.log" 2>&1
  if [ $? -ne 0 ]; then
    echo "FAILED: coaml/${method}_t${TSUFFIX}/${inst} (see /tmp/coaml_${method}_t${TSUFFIX}_${inst}.log)"
    FAILED+=("coaml/${method}_t${TSUFFIX}/${inst}")
  fi
}

echo "=== Threshold ${THRESHOLD} (suffix t${TSUFFIX}) ==="
for inst in $INSTANCES; do run_rh v1final "$V1_MODEL" "$inst"; done
for inst in $INSTANCES; do run_rh v2final "$V2_MODEL" "$inst"; done
for inst in $INSTANCES; do run_coaml v1final "$V1_MODEL" "$inst"; done
for inst in $INSTANCES; do run_coaml v2final "$V2_MODEL" "$inst"; done

echo
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "All 48 threshold-${THRESHOLD} runs completed successfully."
else
  echo "${#FAILED[@]} run(s) FAILED:"
  printf '  %s\n' "${FAILED[@]}"
fi
