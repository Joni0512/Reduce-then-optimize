#!/bin/bash
# Full baseline + v1_final + v2_final comparison (RH + COAML) for an arbitrary
# batch_interval/step_size window, at a given threshold, seed 42 throughout.
# Unlike run_final_compare_threshold.sh, this ALSO reruns the baseline,
# because baseline results depend on the window config (unlike threshold).
#
# Usage: ./run_final_compare_window.sh <batch_interval> <step_size> <threshold>
# Example: ./run_final_compare_window.sh 50 10 0.5
set -u

BATCH_INTERVAL="${1:?usage: run_final_compare_window.sh <batch_interval> <step_size> <threshold>}"
STEP_SIZE="${2:?usage: run_final_compare_window.sh <batch_interval> <step_size> <threshold>}"
THRESHOLD="${3:?usage: run_final_compare_window.sh <batch_interval> <step_size> <threshold>}"
TSUFFIX="${THRESHOLD/./}"

BASE_OUT="new_tests/final_compare/bi${BATCH_INTERVAL}_ss${STEP_SIZE}"

INSTANCES="lc108 lc109 lc207 lc208 lr111 lr112 lr210 lr211 lrc107 lrc108 lrc207 lrc208"

V1_MODEL="outputs/models_v2/rgnn_mixed_c2_pw2_v1_final/rgnn_mixed_c2_pw2_v1_final_best_val_f3.pt"
V2_MODEL="outputs/models_v2_gnnv2/rgnn_mixed_c2_pw2_v2_final/rgnn_mixed_c2_pw2_v2_final_best_val_f3.pt"
NOPRUNE_MODEL="outputs/models_v2_gnnv2/rgnn_mixed_c2_pw10_v2/rgnn_mixed_c2_pw10_v2_best_val_f3.pt"  # unused when pruner is off, kept for log parity

FAILED=()

run_rh() {
  local method="$1" pruner="$2" model="$3" folder="$4" inst="$5"
  local out_dir="${BASE_OUT}/rh_${folder}/${inst}"
  echo ">>> rh/${folder}/${inst}"
  python rtv_solver/main.py \
    --mode offline \
    --input_file "solutions/li_lim/manifests/${inst}.json" \
    --input_dir "solutions/li_lim/manifests/" \
    --imitation_solution_file "solutions/li_lim/manifests/${inst}.json" \
    --use_request_graph_pruner "$pruner" \
    --request_graph_model_path "$model" \
    --request_graph_threshold "$THRESHOLD" \
    --max_cardinality 2 --batch_interval "$BATCH_INTERVAL" --step_size "$STEP_SIZE" \
    --output_dir "$out_dir" \
    > "/tmp/rh_${folder}_${inst}.log" 2>&1
  if [ $? -ne 0 ]; then
    echo "FAILED: rh/${folder}/${inst} (see /tmp/rh_${folder}_${inst}.log)"
    FAILED+=("rh/${folder}/${inst}")
  fi
}

run_coaml() {
  local method="$1" pruner="$2" model="$3" folder="$4" inst="$5"
  local out_dir="${BASE_OUT}/coaml_${folder}/${inst}"
  echo ">>> coaml/${folder}/${inst}"
  python rtv_solver/main.py \
    --mode coaml --input_dir "" \
    --input_file "solutions/li_lim/manifests/${inst}.json" \
    --imitation_solution_file "solutions/li_lim/manifests/${inst}.json" \
    --use_request_graph_pruner "$pruner" \
    --request_graph_model_path "$model" \
    --request_graph_threshold "$THRESHOLD" \
    --max_cardinality 2 --batch_interval "$BATCH_INTERVAL" --step_size "$STEP_SIZE" \
    --output_dir "$out_dir" \
    > "/tmp/coaml_${folder}_${inst}.log" 2>&1
  if [ $? -ne 0 ]; then
    echo "FAILED: coaml/${folder}/${inst} (see /tmp/coaml_${folder}_${inst}.log)"
    FAILED+=("coaml/${folder}/${inst}")
  fi
}

echo "=== Window bi${BATCH_INTERVAL}_ss${STEP_SIZE}, threshold ${THRESHOLD} ==="

for inst in $INSTANCES; do run_rh baseline False "$NOPRUNE_MODEL" "baseline" "$inst"; done
for inst in $INSTANCES; do run_rh v1final True "$V1_MODEL" "v1final_t${TSUFFIX}" "$inst"; done
for inst in $INSTANCES; do run_rh v2final True "$V2_MODEL" "v2final_t${TSUFFIX}" "$inst"; done

for inst in $INSTANCES; do run_coaml baseline False "$NOPRUNE_MODEL" "baseline" "$inst"; done
for inst in $INSTANCES; do run_coaml v1final True "$V1_MODEL" "v1final_t${TSUFFIX}" "$inst"; done
for inst in $INSTANCES; do run_coaml v2final True "$V2_MODEL" "v2final_t${TSUFFIX}" "$inst"; done

echo
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "All 72 runs (bi${BATCH_INTERVAL}_ss${STEP_SIZE}, t${TSUFFIX}) completed successfully."
else
  echo "${#FAILED[@]} run(s) FAILED:"
  printf '  %s\n' "${FAILED[@]}"
fi
