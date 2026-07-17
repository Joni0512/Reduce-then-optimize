#!/bin/bash
# Parametrisierte Version von run_window_50_10_quick.sh: nimmt SEED als
# Kommandozeilenargument, damit jeder Seed als eigener SLURM-Job laeuft
# (robuster + parallelisierbar statt ein Job mit 5 Seeds nacheinander).
# Schreibt in denselben OUT_BASE-Ordner wie der Seed-42-Lauf, damit alle
# Seeds am Ende gemeinsam ausgewertet werden koennen.
set -u

if [ $# -ne 1 ]; then
  echo "Usage: $0 <seed>"
  exit 1
fi
SEED="$1"

INSTANCES="lc108 lc109 lc207 lc208 lr111 lr112 lr210 lr211 lrc107 lrc108 lrc207 lrc208"
STEP_SIZE=10
BATCH_INTERVAL=50
PRUNER_MODEL="outputs/request_pruner_mlp_bi50_ss10/request_pruner_mlp_h64_l3_d0p0_pw1p0_lr0p001/request_pruner_mlp_h64_l3_d0p0_pw1p0_lr0p001_best_val_f3.pt"
THRESHOLDS="0.4 0.5"
OUT_BASE="experiment_window_50_10_quick"

n_inst=$(echo $INSTANCES | wc -w)
n_thr=$(echo $THRESHOLDS | wc -w)
total=$(( n_inst * (1 + n_thr) ))
count=0
skipped=0

already_done() {
  local run_dir="$1"
  compgen -G "${run_dir}/run_offline_mc2_bi${BATCH_INTERVAL}_ss${STEP_SIZE}/*/final/results.json" > /dev/null
}

echo "=== Baseline (seed=$SEED) ==="
for inst in $INSTANCES; do
  count=$((count + 1))
  run_dir="${OUT_BASE}/baseline/${inst}_seed${SEED}"
  if already_done "$run_dir"; then
    skipped=$((skipped + 1))
    echo ">>> [$count/$total] baseline / $inst  (skip - already done)"
    continue
  fi
  echo ">>> [$count/$total] baseline / $inst"
  ./venv/bin/python3 rtv_solver/main.py \
    --mode offline \
    --input_file "solutions/li_lim/manifests/${inst}.json" \
    --input_dir "solutions/li_lim/manifests/" \
    --imitation_solution_file "solutions/li_lim/manifests/${inst}.json" \
    --use_request_graph_pruner False \
    --use_request_pruner False \
    --seed "$SEED" \
    --max_cardinality 2 --batch_interval "$BATCH_INTERVAL" --step_size "$STEP_SIZE" \
    --output_dir "$run_dir" \
    > /dev/null 2>&1
done

for threshold in $THRESHOLDS; do
  echo "=== threshold=${threshold} (seed=$SEED) ==="
  for inst in $INSTANCES; do
    count=$((count + 1))
    run_dir="${OUT_BASE}/thr${threshold}/${inst}_seed${SEED}"
    if already_done "$run_dir"; then
      skipped=$((skipped + 1))
      echo ">>> [$count/$total] thr=${threshold} / $inst  (skip - already done)"
      continue
    fi
    echo ">>> [$count/$total] thr=${threshold} / $inst"
    ./venv/bin/python3 rtv_solver/main.py \
      --mode offline \
      --input_file "solutions/li_lim/manifests/${inst}.json" \
      --input_dir "solutions/li_lim/manifests/" \
      --imitation_solution_file "solutions/li_lim/manifests/${inst}.json" \
      --use_request_graph_pruner False \
      --use_request_pruner True \
      --request_pruner_model_path "$PRUNER_MODEL" \
      --request_pruner_threshold "$threshold" \
      --seed "$SEED" \
      --max_cardinality 2 --batch_interval "$BATCH_INTERVAL" --step_size "$STEP_SIZE" \
      --output_dir "$run_dir" \
      > /dev/null 2>&1
  done
done

echo "=== DONE (seed=$SEED): ${count}/${total} runs attempted, ${skipped} skipped (already done) ==="
