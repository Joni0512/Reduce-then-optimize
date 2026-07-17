#!/bin/bash
# Reruns the COAML leg of the RH/COAML final_compare sweep (baseline, v1_final,
# v2_final @ threshold 0.5) now that the KeyError: 'status' bug in
# stats_parser.py's _compute_request_development is fixed (2026-07-12: it was
# tripping over interleaved "PrunerImitationDiagnostics" log records that lack
# a "status" key). Unlike the first attempt, this uses `set -e` per-run via an
# explicit exit-code check so a crash is reported instead of silently skipped.
set -u

BATCH_INTERVAL=400
STEP_SIZE=100
BASE_OUT="new_tests/final_compare/bi${BATCH_INTERVAL}_ss${STEP_SIZE}"

INSTANCES="lc108 lc109 lc207 lc208 lr111 lr112 lr210 lr211 lrc107 lrc108 lrc207 lrc208"

V1_MODEL="outputs/models_v2/rgnn_mixed_c2_pw2_v1_final/rgnn_mixed_c2_pw2_v1_final_best_val_f3.pt"
V2_MODEL="outputs/models_v2_gnnv2/rgnn_mixed_c2_pw2_v2_final/rgnn_mixed_c2_pw2_v2_final_best_val_f3.pt"

FAILED=()

run_one() {
  local method="$1" pruner="$2" model="$3" inst="$4"
  local out_dir="${BASE_OUT}/${method}/${inst}"
  echo ">>> coaml/${method}/${inst}"
  python rtv_solver/main.py \
    --mode coaml --input_dir "" \
    --input_file "solutions/li_lim/manifests/${inst}.json" \
    --imitation_solution_file "solutions/li_lim/manifests/${inst}.json" \
    --use_request_graph_pruner "$pruner" \
    --request_graph_model_path "$model" \
    --request_graph_threshold 0.5 \
    --max_cardinality 2 --batch_interval "$BATCH_INTERVAL" --step_size "$STEP_SIZE" \
    --output_dir "$out_dir" \
    > /tmp/coaml_${method}_${inst}.log 2>&1
  if [ $? -ne 0 ]; then
    echo "FAILED: coaml/${method}/${inst} (see /tmp/coaml_${method}_${inst}.log)"
    FAILED+=("${method}/${inst}")
  fi
}

for inst in $INSTANCES; do
  run_one "coaml_baseline" False "outputs/models_v2_gnnv2/rgnn_mixed_c2_pw10_v2/rgnn_mixed_c2_pw10_v2_best_val_f3.pt" "$inst"
done
for inst in $INSTANCES; do
  run_one "coaml_v1final_t05" True "$V1_MODEL" "$inst"
done
for inst in $INSTANCES; do
  run_one "coaml_v2final_t05" True "$V2_MODEL" "$inst"
done

echo
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "All 36 COAML runs completed successfully."
else
  echo "${#FAILED[@]} run(s) FAILED:"
  printf '  %s\n' "${FAILED[@]}"
fi
