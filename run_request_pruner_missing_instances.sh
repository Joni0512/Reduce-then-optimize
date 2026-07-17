#!/bin/bash
# 2026-07-14: Phase 0 of the request-pruner design discussion. Generates
# rolling-horizon baseline logs (assignment_data.jsonl) for the 44 li_lim
# instances that don't have them yet - only the 12 pair-pruner "test" split
# instances had these logs (leftovers of earlier pair-pruner eval sweeps
# under outputs/eval_rh_no_learning/). Same config as those 12 runs
# (USE_REQUEST_GRAPH_PRUNER=false, MAX_CARDINALITY=2, BATCH_INTERVAL=400,
# STEP_SIZE=100 - see outputs/eval_rh_no_learning/lc108/rh_no_learning/config.json)
# so all 56 li_lim instances end up with directly comparable logs.
#
# Config.create_output_dir always nests the real output under
# outputs/<output_dir>/run_<mode>_mc<card>_bi<bi>_ss<ss>/<instance>_<timestamp>/
# regardless of what --output_dir is - see rtv_solver/structure/config.py:212-222.
# So after each run this script copies assignment_data.jsonl + config.json out
# of that nested/timestamped path into the SAME flat layout the existing 12
# instances use (outputs/eval_rh_no_learning/<instance>/rh_no_learning/), so
# request_pruner_signal_check.py's path lookup keeps working unchanged for
# all 56 instances.
set -u

INSTANCES="lc101 lc102 lc103 lc104 lc105 lc106 lc107 lc201 lc202 lc203 lc204 lc205 lc206 lr101 lr102 lr103 lr104 lr105 lr106 lr107 lr108 lr109 lr110 lr201 lr202 lr203 lr204 lr205 lr206 lr207 lr208 lr209 lrc101 lrc102 lrc103 lrc104 lrc105 lrc106 lrc201 lrc202 lrc203 lrc204 lrc205 lrc206"

STAGING_OUT="staging_request_pruner_phase0"
FINAL_BASE="outputs/eval_rh_no_learning"
NOPRUNE_MODEL="outputs/models_v2_gnnv2/rgnn_mixed_c2_pw10_v2/rgnn_mixed_c2_pw10_v2_best_val_f3.pt"

count=0
total=$(echo "$INSTANCES" | wc -w | tr -d ' ')

for inst in $INSTANCES; do
  count=$((count + 1))
  echo ">>> [$count/$total] $inst"

  ./venv/bin/python3 rtv_solver/main.py \
    --mode offline \
    --input_file "solutions/li_lim/manifests/${inst}.json" \
    --input_dir "solutions/li_lim/manifests/" \
    --imitation_solution_file "solutions/li_lim/manifests/${inst}.json" \
    --use_request_graph_pruner False \
    --request_graph_model_path "$NOPRUNE_MODEL" \
    --request_graph_threshold 0.5 \
    --seed 42 \
    --max_cardinality 2 --batch_interval 400 --step_size 100 \
    --output_dir "${STAGING_OUT}/${inst}"

  # Config nests the real run under a run_offline_mc2_bi400_ss100/<inst>_<timestamp>/
  # subfolder - find it (there's exactly one per instance since STAGING_OUT is unique
  # per instance) and copy the two files we need into the flat target layout.
  run_dir=$(find "outputs/${STAGING_OUT}/${inst}" -maxdepth 2 -type d -name "${inst}_*" | head -1)
  if [ -z "$run_dir" ]; then
    echo "!!! could not find output dir for $inst, skipping copy"
    continue
  fi

  target_dir="${FINAL_BASE}/${inst}/rh_no_learning"
  mkdir -p "$target_dir"
  cp "${run_dir}/assignment_data.jsonl" "$target_dir/assignment_data.jsonl"
  cp "${run_dir}/config.json" "$target_dir/config.json"
  echo "    -> ${target_dir}/assignment_data.jsonl ($(wc -l < "${run_dir}/assignment_data.jsonl") windows)"
done

echo "=== DONE: ${count}/${total} instances processed ==="
