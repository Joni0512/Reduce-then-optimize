#!/bin/bash
# 2026-07-16: threshold-parameterized version of run_rho_pruner_comparison_12.sh
# (RHO ohne Learning, --mode offline). Skips baseline (threshold-independent,
# already computed by the t=0.5 run) and tags output dirs with the threshold
# so 0.4/0.5/0.6 results coexist instead of overwriting each other - same
# fix as run_opt_pruner_comparison_12_threshold.sh needed for the OPT sweep.
#
# Usage: ./run_rho_pruner_comparison_12_threshold.sh <threshold>
set -u

THRESHOLD="${1:?usage: run_rho_pruner_comparison_12_threshold.sh <threshold>}"
INSTANCES="lc108 lc109 lc207 lc208 lr111 lr112 lr210 lr211 lrc107 lrc108 lrc207 lrc208"
BATCH_INTERVAL=200
STEP_SIZE=100
REQUEST_PRUNER_MODEL="outputs/request_pruner_mlp_bi200_ss100/request_pruner_mlp_h32_l1_d0p2_pw1p0_lr0p01/request_pruner_mlp_h32_l1_d0p2_pw1p0_lr0p01_best_val_f3.pt"
OUT_BASE="outputs/rho_single_instance/bi200_ss100"
TSUFFIX="${THRESHOLD/./p}"

total=$(( $(echo $INSTANCES | wc -w) * 3 ))
count=0

run_one() {
  local variant="$1" inst="$2" pruner_flags="$3"
  count=$((count + 1))
  echo ">>> [$count/$total] $variant / $inst (t=$THRESHOLD)"
  ./venv/bin/python3 rtv_solver/main.py \
    --mode offline \
    --input_file "solutions/li_lim/manifests/${inst}.json" \
    --input_dir "solutions/li_lim/manifests/" \
    --imitation_solution_file "solutions/li_lim/manifests/${inst}.json" \
    $pruner_flags \
    --seed 42 \
    --max_cardinality 2 --batch_interval "$BATCH_INTERVAL" --step_size "$STEP_SIZE" \
    --output_dir "${OUT_BASE}/${inst}/${variant}_t${TSUFFIX}" \
    > "/tmp/rho_cmp_t${THRESHOLD}_${variant}_${inst}.log" 2>&1
}

for inst in $INSTANCES; do
  run_one "request_pruner" "$inst" "--use_request_pruner True --request_pruner_model_path $REQUEST_PRUNER_MODEL --request_pruner_threshold $THRESHOLD --use_request_graph_pruner False"
  run_one "pair_pruner" "$inst" "--use_request_pruner False --use_request_graph_pruner True --request_graph_threshold $THRESHOLD"
  run_one "both_pruners" "$inst" "--use_request_pruner True --request_pruner_model_path $REQUEST_PRUNER_MODEL --request_pruner_threshold $THRESHOLD --use_request_graph_pruner True --request_graph_threshold $THRESHOLD"
done

echo "=== DONE: ${count}/${total} runs attempted (RHO no-learning, threshold=$THRESHOLD) ==="
