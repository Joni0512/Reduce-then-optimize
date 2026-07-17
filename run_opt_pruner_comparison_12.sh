#!/bin/bash
# 2026-07-16: OPT-mode (COAML mode="optimal") pruner comparison across all 12
# test instances, at bi200_ss100, threshold 0.5 for both pruners. Four
# variants per instance: baseline (no pruner), request pruner alone, pair
# (request-request) pruner alone, both combined - same structure as the
# lc108/lr111 spot checks discussed in chat, just scaled to the full test set
# so the runtime/service-rate comparison isn't based on 2 instances only.
set -u

INSTANCES="lc108 lc109 lc207 lc208 lr111 lr112 lr210 lr211 lrc107 lrc108 lrc207 lrc208"
BATCH_INTERVAL=200
STEP_SIZE=100
REQUEST_PRUNER_MODEL="outputs/request_pruner_mlp_bi200_ss100/request_pruner_mlp_h32_l1_d0p2_pw1p0_lr0p01/request_pruner_mlp_h32_l1_d0p2_pw1p0_lr0p01_best_val_f3.pt"
THRESHOLD=0.5

total=$(( $(echo $INSTANCES | wc -w) * 4 ))
count=0

for inst in $INSTANCES; do
  count=$((count + 1))
  echo ">>> [$count/$total] baseline / $inst"
  ./venv/bin/python3 rtv_solver/pipeline/run_opt_single_instance.py \
    --instance "$inst" --batch_interval "$BATCH_INTERVAL" --step_size "$STEP_SIZE" \
    --use_request_pruner False --use_request_graph_pruner False \
    > "/tmp/opt_cmp_baseline_${inst}.log" 2>&1

  count=$((count + 1))
  echo ">>> [$count/$total] request_pruner / $inst"
  ./venv/bin/python3 rtv_solver/pipeline/run_opt_single_instance.py \
    --instance "$inst" --batch_interval "$BATCH_INTERVAL" --step_size "$STEP_SIZE" \
    --use_request_pruner True --request_pruner_model_path "$REQUEST_PRUNER_MODEL" --request_pruner_threshold "$THRESHOLD" \
    --use_request_graph_pruner False \
    > "/tmp/opt_cmp_reqpruner_${inst}.log" 2>&1

  count=$((count + 1))
  echo ">>> [$count/$total] pair_pruner / $inst"
  ./venv/bin/python3 rtv_solver/pipeline/run_opt_single_instance.py \
    --instance "$inst" --batch_interval "$BATCH_INTERVAL" --step_size "$STEP_SIZE" \
    --use_request_pruner False \
    --use_request_graph_pruner True --request_graph_threshold "$THRESHOLD" \
    > "/tmp/opt_cmp_pairpruner_${inst}.log" 2>&1

  count=$((count + 1))
  echo ">>> [$count/$total] both_pruners / $inst"
  ./venv/bin/python3 rtv_solver/pipeline/run_opt_single_instance.py \
    --instance "$inst" --batch_interval "$BATCH_INTERVAL" --step_size "$STEP_SIZE" \
    --use_request_pruner True --request_pruner_model_path "$REQUEST_PRUNER_MODEL" --request_pruner_threshold "$THRESHOLD" \
    --use_request_graph_pruner True --request_graph_threshold "$THRESHOLD" \
    > "/tmp/opt_cmp_both_${inst}.log" 2>&1
done

echo "=== DONE: ${count}/${total} runs attempted ==="
