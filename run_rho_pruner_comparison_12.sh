#!/bin/bash
# 2026-07-16: RHO-no-learning (--mode offline, plain myopic ILP, no COAML/
# imitation-learning component) counterpart to run_opt_pruner_comparison_12.sh
# - same 12 instances, same bi200_ss100 window, same threshold 0.5 for both
# pruners, same 4 variants (baseline / request_pruner / pair_pruner / both).
# Goal: check whether the catastrophic service-rate collapse found against
# the OPT ceiling at t=0.5 also happens against the plain RHO baseline, or
# whether RHO's graceful ILP fallback (vs. OPT's brittle all-or-nothing
# y*-matching, see coaml_pipeline.py notes) makes it more robust.
#
# Unlike run_opt_single_instance.py, main.py's offline mode already measures
# and saves real wall-clock runtime into results.json's stats.total_time
# (see main.py:206/262/264) - no custom timing wrapper needed here.
set -u

INSTANCES="lc108 lc109 lc207 lc208 lr111 lr112 lr210 lr211 lrc107 lrc108 lrc207 lrc208"
BATCH_INTERVAL=200
STEP_SIZE=100
REQUEST_PRUNER_MODEL="outputs/request_pruner_mlp_bi200_ss100/request_pruner_mlp_h32_l1_d0p2_pw1p0_lr0p01/request_pruner_mlp_h32_l1_d0p2_pw1p0_lr0p01_best_val_f3.pt"
THRESHOLD=0.5
OUT_BASE="outputs/rho_single_instance/bi200_ss100"

total=$(( $(echo $INSTANCES | wc -w) * 4 ))
count=0

run_one() {
  local variant="$1" inst="$2" pruner_flags="$3"
  count=$((count + 1))
  echo ">>> [$count/$total] $variant / $inst"
  ./venv/bin/python3 rtv_solver/main.py \
    --mode offline \
    --input_file "solutions/li_lim/manifests/${inst}.json" \
    --input_dir "solutions/li_lim/manifests/" \
    --imitation_solution_file "solutions/li_lim/manifests/${inst}.json" \
    $pruner_flags \
    --seed 42 \
    --max_cardinality 2 --batch_interval "$BATCH_INTERVAL" --step_size "$STEP_SIZE" \
    --output_dir "${OUT_BASE}/${inst}/${variant}" \
    > "/tmp/rho_cmp_${variant}_${inst}.log" 2>&1
}

for inst in $INSTANCES; do
  run_one "baseline" "$inst" "--use_request_pruner False --use_request_graph_pruner False"
  run_one "request_pruner" "$inst" "--use_request_pruner True --request_pruner_model_path $REQUEST_PRUNER_MODEL --request_pruner_threshold $THRESHOLD --use_request_graph_pruner False"
  run_one "pair_pruner" "$inst" "--use_request_pruner False --use_request_graph_pruner True --request_graph_threshold $THRESHOLD"
  run_one "both_pruners" "$inst" "--use_request_pruner True --request_pruner_model_path $REQUEST_PRUNER_MODEL --request_pruner_threshold $THRESHOLD --use_request_graph_pruner True --request_graph_threshold $THRESHOLD"
done

echo "=== DONE: ${count}/${total} runs attempted (RHO no-learning, threshold=$THRESHOLD) ==="
