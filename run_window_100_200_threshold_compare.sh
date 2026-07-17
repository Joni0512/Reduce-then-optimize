#!/bin/bash
# 2026-07-15: threshold comparison at the 200/100 window config, using the
# window-specific model + safety floor (min_retention_fraction=0.3, fixed
# default in RequestPruner). Baseline is threshold-independent and already
# collected in outputs/experiment_window_100_200_v2_floor/baseline/ - only
# the request_pruner variant is rerun here, once per threshold.
set -u

INSTANCES="lc108 lc109 lc207 lc208 lr111 lr112 lr210 lr211 lrc107 lrc108 lrc207 lrc208"
SEEDS="42 1 2 3 4"
STEP_SIZE=100
BATCH_INTERVAL=200
PRUNER_MODEL="outputs/request_pruner_mlp_bi200_ss100/request_pruner_mlp_h32_l1_d0p2_pw1p0_lr0p01/request_pruner_mlp_h32_l1_d0p2_pw1p0_lr0p01_best_val_f3.pt"
THRESHOLDS="0.4 0.6"

total=$(( $(echo $INSTANCES | wc -w) * $(echo $SEEDS | wc -w) * $(echo $THRESHOLDS | wc -w) ))
count=0

for threshold in $THRESHOLDS; do
  OUT_BASE="experiment_window_100_200_thr${threshold}"
  for inst in $INSTANCES; do
    for seed in $SEEDS; do
      count=$((count + 1))
      echo ">>> [$count/$total] thr=${threshold} / $inst / seed=$seed"

      ./venv/bin/python3 rtv_solver/main.py \
        --mode offline \
        --input_file "solutions/li_lim/manifests/${inst}.json" \
        --input_dir "solutions/li_lim/manifests/" \
        --imitation_solution_file "solutions/li_lim/manifests/${inst}.json" \
        --use_request_graph_pruner False \
        --use_request_pruner True \
        --request_pruner_model_path "$PRUNER_MODEL" \
        --request_pruner_threshold "$threshold" \
        --seed "$seed" \
        --max_cardinality 2 --batch_interval "$BATCH_INTERVAL" --step_size "$STEP_SIZE" \
        --output_dir "${OUT_BASE}/request_pruner/${inst}_seed${seed}" \
        > /dev/null 2>&1
    done
  done
done

echo "=== DONE: ${count}/${total} runs ==="
