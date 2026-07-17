#!/bin/bash
# 2026-07-14: Phase 4 request-pruner check, window config "200-100" (thesis
# notation: horizon-step = BATCH_INTERVAL-STEP_SIZE), i.e. STEP_SIZE=100,
# BATCH_INTERVAL=200 - matches Jasper Ihno Wiltfang's master's thesis
# ("Structured Learning for the PDPTW", Section 6.1 / Figure 10 / Table 4,
# the "100-200" config there). Runs baseline (no pruner) vs request-pruner-
# enabled, 5 seeds each, across all 12 test-set instances - same instances
# the thesis itself uses for validation (Figure 8/9, Table 4).
set -u

INSTANCES="lc108 lc109 lc207 lc208 lr111 lr112 lr210 lr211 lrc107 lrc108 lrc207 lrc208"
SEEDS="42 1 2 3 4"
STEP_SIZE=100
BATCH_INTERVAL=200
PRUNER_MODEL="outputs/request_pruner_mlp_bi200_ss100/request_pruner_mlp_h32_l1_d0p2_pw1p0_lr0p01/request_pruner_mlp_h32_l1_d0p2_pw1p0_lr0p01_best_val_f3.pt"
OUT_BASE="experiment_window_100_200_v2_floor"

total=$(( $(echo $INSTANCES | wc -w) * $(echo $SEEDS | wc -w) * 2 ))
count=0

for variant in baseline request_pruner; do
  for inst in $INSTANCES; do
    for seed in $SEEDS; do
      count=$((count + 1))
      echo ">>> [$count/$total] $variant / $inst / seed=$seed"

      if [ "$variant" = "request_pruner" ]; then
        pruner_flags="--use_request_pruner True --request_pruner_model_path $PRUNER_MODEL --request_pruner_threshold 0.3"
      else
        pruner_flags="--use_request_pruner False"
      fi

      ./venv/bin/python3 rtv_solver/main.py \
        --mode offline \
        --input_file "solutions/li_lim/manifests/${inst}.json" \
        --input_dir "solutions/li_lim/manifests/" \
        --imitation_solution_file "solutions/li_lim/manifests/${inst}.json" \
        --use_request_graph_pruner False \
        $pruner_flags \
        --seed "$seed" \
        --max_cardinality 2 --batch_interval "$BATCH_INTERVAL" --step_size "$STEP_SIZE" \
        --output_dir "${OUT_BASE}/${variant}/${inst}_seed${seed}" \
        > /dev/null 2>&1
    done
  done
done

echo "=== DONE: ${count}/${total} runs ==="
