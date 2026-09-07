#!/bin/bash
set -e

# 2026-09-07: RHO-without-learning baseline for the 7 instances from our
# balanced 12-instance SRL test set that are NOT already in Table 4
# (RHO/SIL/OPT paper comparison) at horizon=200/step=100/cardinality=2 -
# see chat. Table 4 already covers lc108, lc109, lr111, lr112, lrc107,
# lrc108 at this exact config (RHO avg 78.0%); this fills in the remaining
# 7 instances from our SRL set (lc107, lc207, lc208, lr203, lr210, lrc201,
# lrc207) for a complete, apples-to-apples RHO comparison against SRL/GAT.
# Same flag pattern as run_rh_no_learning_mc3_h200s100.sh (existing
# reference script), just cardinality=2 and no pruner.

CARDINALITY=2
BATCH_INTERVAL=200
STEP_SIZE=100

INSTANCES=(
  lc107 lc207 lc208
  lr203 lr210
  lrc201 lrc207
)

for INSTANCE in "${INSTANCES[@]}"; do
  FILE="solutions/li_lim/manifests/${INSTANCE}.json"
  echo "===== Running $INSTANCE | RH/offline no pruner | mc=${CARDINALITY} | h${BATCH_INTERVAL}s${STEP_SIZE} ====="
  python3 -m rtv_solver.main \
    --mode offline \
    --input_dir "" \
    --input_file "$FILE" \
    --use_request_graph_pruner False \
    --max_cardinality "$CARDINALITY" \
    --batch_interval "$BATCH_INTERVAL" \
    --step_size "$STEP_SIZE" \
    --output_dir "outputs/eval_rh_no_learning_baseline_mc${CARDINALITY}_h${BATCH_INTERVAL}s${STEP_SIZE}/${INSTANCE}"
done

echo "DONE RH/offline baseline (mc=2, horizon 200-100) for SRL-set gap instances"
