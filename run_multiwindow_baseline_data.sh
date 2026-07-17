#!/bin/bash
# 2026-07-14: Window-size robustness project (request pruner). Generates
# rolling-horizon baseline logs (assignment_data.jsonl, USE_REQUEST_GRAPH_PRUNER=false,
# USE_REQUEST_PRUNER=false) for all 56 li_lim instances at THREE additional
# window configs, on top of the existing BATCH_INTERVAL=400/STEP_SIZE=100
# data already in outputs/eval_rh_no_learning/ (Phase 0 original). This is
# needed because the current request pruner was trained ONLY on 400/100
# data and was found (2026-07-14, lr211/lrc208 investigation) to extrapolate
# catastrophically at other window sizes - requests_per_vehicle values it
# never saw during training (z-score up to +2.9) collapsed its scores to
# near-zero. Training on the union of multiple window configs should expose
# it to a much wider feature range.
#
# Each config gets its own top-level output folder, named the same way as
# the original (eval_rh_no_learning) but suffixed with bi<BATCH_INTERVAL>_ss<STEP_SIZE>,
# so all 4 window configs' data can coexist without overwriting each other:
#   outputs/eval_rh_no_learning/                  bi400_ss100 (already exists, Phase 0)
#   outputs/eval_rh_no_learning_bi200_ss100/       new
#   outputs/eval_rh_no_learning_bi400_ss200/       new
#   outputs/eval_rh_no_learning_bi50_ss10/         new
set -u

INSTANCES="lc101 lc102 lc103 lc104 lc105 lc106 lc107 lc108 lc109 lc201 lc202 lc203 lc204 lc205 lc206 lc207 lc208 lr101 lr102 lr103 lr104 lr105 lr106 lr107 lr108 lr109 lr110 lr111 lr112 lr201 lr202 lr203 lr204 lr205 lr206 lr207 lr208 lr209 lr210 lr211 lrc101 lrc102 lrc103 lrc104 lrc105 lrc106 lrc107 lrc108 lrc201 lrc202 lrc203 lrc204 lrc205 lrc206 lrc207 lrc208"

# (BATCH_INTERVAL, STEP_SIZE) pairs to generate - the 3 missing window configs.
CONFIGS="200:100 400:200 50:10"

total=$(( $(echo $INSTANCES | wc -w) * 3 ))
count=0

for cfg in $CONFIGS; do
  BI="${cfg%%:*}"
  SS="${cfg##*:}"
  STAGING_OUT="staging_multiwindow_bi${BI}_ss${SS}"
  FINAL_BASE="outputs/eval_rh_no_learning_bi${BI}_ss${SS}"

  for inst in $INSTANCES; do
    count=$((count + 1))
    echo ">>> [$count/$total] bi=${BI} ss=${SS} / $inst"

    ./venv/bin/python3 rtv_solver/main.py \
      --mode offline \
      --input_file "solutions/li_lim/manifests/${inst}.json" \
      --input_dir "solutions/li_lim/manifests/" \
      --imitation_solution_file "solutions/li_lim/manifests/${inst}.json" \
      --use_request_graph_pruner False \
      --use_request_pruner False \
      --seed 42 \
      --max_cardinality 2 --batch_interval "$BI" --step_size "$SS" \
      --output_dir "${STAGING_OUT}/${inst}" \
      > /dev/null 2>&1

    run_dir=$(find "outputs/${STAGING_OUT}/${inst}" -maxdepth 2 -type d -name "${inst}_*" | head -1)
    if [ -z "$run_dir" ]; then
      echo "!!! could not find output dir for bi=${BI} ss=${SS} $inst, skipping copy"
      continue
    fi

    target_dir="${FINAL_BASE}/${inst}/rh_no_learning"
    mkdir -p "$target_dir"
    cp "${run_dir}/assignment_data.jsonl" "$target_dir/assignment_data.jsonl"
    cp "${run_dir}/config.json" "$target_dir/config.json"
  done

  rm -rf "outputs/${STAGING_OUT}"
done

echo "=== DONE: ${count}/${total} instance-config combinations processed ==="
