#!/bin/bash
# 2026-07-15: 400/200 window config ("vierhundert zweihundert", groesseres
# Fenster/seltenere Re-Optimierung). Baseline once (threshold-independent),
# then the window-specific model at thresholds 0.3-0.6, 12 test instances x
# 5 seeds each.
set -u

INSTANCES="lc108 lc109 lc207 lc208 lr111 lr112 lr210 lr211 lrc107 lrc108 lrc207 lrc208"
SEEDS="42 1 2 3 4"
STEP_SIZE=200
BATCH_INTERVAL=400
PRUNER_MODEL="outputs/request_pruner_mlp_bi400_ss200/request_pruner_mlp_h32_l1_d0p0_pw1p0_lr0p03/request_pruner_mlp_h32_l1_d0p0_pw1p0_lr0p03_best_val_f3.pt"
THRESHOLDS="0.3 0.4 0.5 0.6"
OUT_BASE="experiment_window_400_200"

n_inst=$(echo $INSTANCES | wc -w)
n_seed=$(echo $SEEDS | wc -w)
n_thr=$(echo $THRESHOLDS | wc -w)
total=$(( n_inst * n_seed * (1 + n_thr) ))
count=0

echo "=== Baseline ==="
for inst in $INSTANCES; do
  for seed in $SEEDS; do
    count=$((count + 1))
    echo ">>> [$count/$total] baseline / $inst / seed=$seed"
    ./venv/bin/python3 rtv_solver/main.py \
      --mode offline \
      --input_file "solutions/li_lim/manifests/${inst}.json" \
      --input_dir "solutions/li_lim/manifests/" \
      --imitation_solution_file "solutions/li_lim/manifests/${inst}.json" \
      --use_request_graph_pruner False \
      --use_request_pruner False \
      --seed "$seed" \
      --max_cardinality 2 --batch_interval "$BATCH_INTERVAL" --step_size "$STEP_SIZE" \
      --output_dir "${OUT_BASE}/baseline/${inst}_seed${seed}" \
      > /dev/null 2>&1
  done
done

for threshold in $THRESHOLDS; do
  echo "=== threshold=${threshold} ==="
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
        --output_dir "${OUT_BASE}/thr${threshold}/${inst}_seed${seed}" \
        > /dev/null 2>&1
    done
  done
done

echo "=== DONE: ${count}/${total} runs ==="
