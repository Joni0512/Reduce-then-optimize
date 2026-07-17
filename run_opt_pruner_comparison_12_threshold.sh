#!/bin/bash
# 2026-07-16: same 12-instance OPT-mode pruner comparison as
# run_opt_pruner_comparison_12.sh, parameterized by threshold so multiple
# thresholds can be swept without colliding (run_opt_single_instance.py now
# tags output dirs with the threshold value). Skips the baseline variant -
# baseline never uses either pruner, so it's threshold-independent and
# already computed by the first (0.5) sweep.
#
# Usage: ./run_opt_pruner_comparison_12_threshold.sh <threshold>
set -u

THRESHOLD="${1:?usage: run_opt_pruner_comparison_12_threshold.sh <threshold>}"
INSTANCES="lc108 lc109 lc207 lc208 lr111 lr112 lr210 lr211 lrc107 lrc108 lrc207 lrc208"
BATCH_INTERVAL=200
STEP_SIZE=100
REQUEST_PRUNER_MODEL="outputs/request_pruner_mlp_bi200_ss100/request_pruner_mlp_h32_l1_d0p2_pw1p0_lr0p01/request_pruner_mlp_h32_l1_d0p2_pw1p0_lr0p01_best_val_f3.pt"

total=$(( $(echo $INSTANCES | wc -w) * 3 ))
count=0

for inst in $INSTANCES; do
  count=$((count + 1))
  echo ">>> [$count/$total] request_pruner t=$THRESHOLD / $inst"
  ./venv/bin/python3 rtv_solver/pipeline/run_opt_single_instance.py \
    --instance "$inst" --batch_interval "$BATCH_INTERVAL" --step_size "$STEP_SIZE" \
    --use_request_pruner True --request_pruner_model_path "$REQUEST_PRUNER_MODEL" --request_pruner_threshold "$THRESHOLD" \
    --use_request_graph_pruner False \
    > "/tmp/opt_cmp_t${THRESHOLD}_reqpruner_${inst}.log" 2>&1

  count=$((count + 1))
  echo ">>> [$count/$total] pair_pruner t=$THRESHOLD / $inst"
  ./venv/bin/python3 rtv_solver/pipeline/run_opt_single_instance.py \
    --instance "$inst" --batch_interval "$BATCH_INTERVAL" --step_size "$STEP_SIZE" \
    --use_request_pruner False \
    --use_request_graph_pruner True --request_graph_threshold "$THRESHOLD" \
    > "/tmp/opt_cmp_t${THRESHOLD}_pairpruner_${inst}.log" 2>&1

  count=$((count + 1))
  echo ">>> [$count/$total] both_pruners t=$THRESHOLD / $inst"
  ./venv/bin/python3 rtv_solver/pipeline/run_opt_single_instance.py \
    --instance "$inst" --batch_interval "$BATCH_INTERVAL" --step_size "$STEP_SIZE" \
    --use_request_pruner True --request_pruner_model_path "$REQUEST_PRUNER_MODEL" --request_pruner_threshold "$THRESHOLD" \
    --use_request_graph_pruner True --request_graph_threshold "$THRESHOLD" \
    > "/tmp/opt_cmp_t${THRESHOLD}_both_${inst}.log" 2>&1
done

echo "=== DONE: ${count}/${total} runs attempted (threshold=$THRESHOLD) ==="
