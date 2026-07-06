#!/bin/bash
set -e

MODEL="outputs/models_v2/rgnn_mixed_all_pw1_v2/rgnn_mixed_all_pw1_v2_best_val_f3.pt"
THRESHOLD=0.5
CARDINALITY=3
BATCH_INTERVAL=50
STEP_SIZE=10

INSTANCES=(
  lc108 lc109 lc207 lc208
  lr111 lr112 lr210 lr211
  lrc107 lrc108 lrc207 lrc208
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

for INSTANCE in "${INSTANCES[@]}"; do
  FILE="solutions/li_lim/manifests/${INSTANCE}.json"
  echo "===== Running $INSTANCE | RH/offline + pruner | mc=${CARDINALITY} | h${BATCH_INTERVAL}s${STEP_SIZE} ====="
  python3 -m rtv_solver.main \
    --mode offline \
    --input_dir "" \
    --input_file "$FILE" \
    --use_request_graph_pruner True \
    --request_graph_model_path "$MODEL" \
    --request_graph_threshold "$THRESHOLD" \
    --max_cardinality "$CARDINALITY" \
    --batch_interval "$BATCH_INTERVAL" \
    --step_size "$STEP_SIZE" \
    --output_dir "outputs/eval_rh_no_learning_pruner_mc${CARDINALITY}_h${BATCH_INTERVAL}s${STEP_SIZE}/${INSTANCE}"
done

echo "DONE RH/offline baseline+pruner evaluation (mc=3, horizon 50-10)"
