#!/bin/bash
set -e

INSTANCES=(
  lc108 lc109 lc207 lc208
  lr111 lr112 lr210 lr211
  lrc107 lrc108 lrc207 lrc208
)

for INSTANCE in "${INSTANCES[@]}"; do
  FILE="solutions/li_lim/manifests/${INSTANCE}.json"

  echo "===== Running $INSTANCE | baseline no pruner ====="

  python3 -m rtv_solver.main \
    --mode coaml \
    --epochs 1 \
    --input_dir "" \
    --input_file "$FILE" \
    --imitation_solution_file "$FILE" \
    --use_request_graph_pruner False \
    --max_cardinality 2 \
    --batch_interval 400 \
    --step_size 100 \
    --output_dir "outputs/eval_baseline/${INSTANCE}"
done

echo "DONE baseline evaluation"
