#!/bin/bash
# 2026-07-14: RequestPruner (request-only pruner) vs. no-pruner comparison,
# threshold sweep 0.1-0.5, RH (offline) + COAML, at a given window config.
# Pair pruner (request_graph_pruner) is kept OFF throughout, to isolate the
# pure effect of the new request-only pruner (decision made 2026-07-14: test
# it in isolation first, not stacked on top of the pair pruner).
#
# Reuses the existing no-pruner baseline runs under
# outputs/new_tests/final_compare/bi${BI}_ss${SS}/{rh,coaml}_baseline instead
# of rerunning them - those runs predate USE_REQUEST_PRUNER entirely (added
# 2026-07-14), so they load with USE_REQUEST_PRUNER defaulting to False, same
# as passing --use_request_pruner False explicitly. Only requires that these
# baseline folders already exist for the given BI/SS (true for bi400_ss100).
#
# Output layout mirrors run_final_compare_window.sh's convention so
# parse_final_compare.py's read()/latest() helpers work unchanged:
#   outputs/new_tests/final_compare/bi${BI}_ss${SS}/rh_reqpruner_t${T}/${inst}
#   outputs/new_tests/final_compare/bi${BI}_ss${SS}/coaml_reqpruner_t${T}/${inst}
#
# Usage: ./run_request_pruner_threshold_compare.sh <batch_interval> <step_size> <model_path>
# Example: ./run_request_pruner_threshold_compare.sh 400 100 outputs/request_pruner_mlp/request_pruner_mlp_h32_l1_d0p0_pw1p0/request_pruner_mlp_h32_l1_d0p0_pw1p0_best_val_f3.pt
set -u

BATCH_INTERVAL="${1:?usage: run_request_pruner_threshold_compare.sh <batch_interval> <step_size> <model_path>}"
STEP_SIZE="${2:?usage: run_request_pruner_threshold_compare.sh <batch_interval> <step_size> <model_path>}"
MODEL="${3:?usage: run_request_pruner_threshold_compare.sh <batch_interval> <step_size> <model_path>}"

BASE_OUT="outputs/new_tests/final_compare/bi${BATCH_INTERVAL}_ss${STEP_SIZE}"
INSTANCES="lc108 lc109 lc207 lc208 lr111 lr112 lr210 lr211 lrc107 lrc108 lrc207 lrc208"
THRESHOLDS="0.1 0.2 0.3 0.4 0.5"

FAILED=()

run_one() {
  local mode="$1" folder_prefix="$2" threshold="$3" inst="$4"
  local tsuffix="${threshold/./}"
  local out_dir="${BASE_OUT}/${folder_prefix}_reqpruner_t${tsuffix}/${inst}"
  echo ">>> ${folder_prefix}/reqpruner_t${threshold}/${inst}"
  local extra_args=()
  if [ "$mode" = "coaml" ]; then
    extra_args=(--mode coaml --input_dir "")
  else
    extra_args=(--mode offline --input_dir "solutions/li_lim/manifests/")
  fi
  ./venv/bin/python3 rtv_solver/main.py \
    "${extra_args[@]}" \
    --input_file "solutions/li_lim/manifests/${inst}.json" \
    --imitation_solution_file "solutions/li_lim/manifests/${inst}.json" \
    --use_request_graph_pruner False \
    --use_request_pruner True \
    --request_pruner_model_path "$MODEL" \
    --request_pruner_threshold "$threshold" \
    --seed 42 \
    --max_cardinality 2 --batch_interval "$BATCH_INTERVAL" --step_size "$STEP_SIZE" \
    --output_dir "$out_dir" \
    > "/tmp/reqpruner_${folder_prefix}_t${tsuffix}_${inst}.log" 2>&1
  if [ $? -ne 0 ]; then
    echo "FAILED: ${folder_prefix}/reqpruner_t${threshold}/${inst} (see /tmp/reqpruner_${folder_prefix}_t${tsuffix}_${inst}.log)"
    FAILED+=("${folder_prefix}/reqpruner_t${threshold}/${inst}")
  fi
}

total=$(( $(echo $THRESHOLDS | wc -w) * $(echo $INSTANCES | wc -w) * 2 ))
count=0

for threshold in $THRESHOLDS; do
  for inst in $INSTANCES; do
    count=$((count + 1))
    echo "[$count/$total]"
    run_one "offline" "rh" "$threshold" "$inst"
  done
done
for threshold in $THRESHOLDS; do
  for inst in $INSTANCES; do
    count=$((count + 1))
    echo "[$count/$total]"
    run_one "coaml" "coaml" "$threshold" "$inst"
  done
done

echo
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "All ${total} request-pruner runs completed successfully."
else
  echo "${#FAILED[@]} run(s) FAILED:"
  printf '  %s\n' "${FAILED[@]}"
fi
