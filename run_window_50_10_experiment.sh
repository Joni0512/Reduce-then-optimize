#!/bin/bash
# BATCH_INTERVAL=50, STEP_SIZE=10 ("50/10", kleinstes Fenster aus der
# Masterarbeit), mehrere Thresholds. Nutzt das eigens dafür trainierte
# Modell (outputs/request_pruner_mlp_bi50_ss10/...).
#
# 2026-07-16: resume-fähig gemacht, nachdem der erste Cluster-Job nach
# --time=02:00:00 nur 23/60 Baseline-Läufe geschafft hatte (50/10 ist die
# mit Abstand langsamste Config, siehe fruehere Timeouts bei der lokalen
# Datengenerierung). Vor jedem Lauf wird geprüft, ob bereits eine
# results.json für genau diese (variant, instance, seed)-Kombination
# existiert - falls ja, wird der Lauf übersprungen statt neu gestartet.
# Dadurch kann derselbe sbatch-Job (oder ein neuer mit laengerem --time)
# beliebig oft resubmitted werden, ohne bereits fertige Läufe zu wiederholen.
set -u

INSTANCES="lc108 lc109 lc207 lc208 lr111 lr112 lr210 lr211 lrc107 lrc108 lrc207 lrc208"
SEEDS="42 1 2 3 4"
STEP_SIZE=10
BATCH_INTERVAL=50
PRUNER_MODEL="outputs/request_pruner_mlp_bi50_ss10/request_pruner_mlp_h64_l3_d0p0_pw1p0_lr0p001/request_pruner_mlp_h64_l3_d0p0_pw1p0_lr0p001_best_val_f3.pt"
THRESHOLDS="0.3 0.4 0.5 0.6"
OUT_BASE="experiment_window_50_10"

n_inst=$(echo $INSTANCES | wc -w)
n_seed=$(echo $SEEDS | wc -w)
n_thr=$(echo $THRESHOLDS | wc -w)
total=$(( n_inst * n_seed * (1 + n_thr) ))
count=0
skipped=0

# Returns 0 (true) if a results.json already exists for this run dir.
already_done() {
  local run_dir="$1"
  compgen -G "${run_dir}/run_offline_mc2_bi${BATCH_INTERVAL}_ss${STEP_SIZE}/*/final/results.json" > /dev/null
}

echo "=== Baseline ==="
for inst in $INSTANCES; do
  for seed in $SEEDS; do
    count=$((count + 1))
    run_dir="${OUT_BASE}/baseline/${inst}_seed${seed}"
    if already_done "$run_dir"; then
      skipped=$((skipped + 1))
      echo ">>> [$count/$total] baseline / $inst / seed=$seed  (skip - already done)"
      continue
    fi
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
      --output_dir "$run_dir" \
      > /dev/null 2>&1
  done
done

for threshold in $THRESHOLDS; do
  echo "=== threshold=${threshold} ==="
  for inst in $INSTANCES; do
    for seed in $SEEDS; do
      count=$((count + 1))
      run_dir="${OUT_BASE}/thr${threshold}/${inst}_seed${seed}"
      if already_done "$run_dir"; then
        skipped=$((skipped + 1))
        echo ">>> [$count/$total] thr=${threshold} / $inst / seed=$seed  (skip - already done)"
        continue
      fi
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
        --output_dir "$run_dir" \
        > /dev/null 2>&1
    done
  done
done

echo "=== DONE: ${count}/${total} runs attempted, ${skipped} skipped (already done) ==="
