#!/bin/bash
# 2026-07-17: SIL (mode="train"+eval) full sweep, SECOND seed (1, not 42),
# all 3 thresholds (0.4/0.5/0.6), bi400/ss200. Runs fully automatically,
# unattended - baseline once (seed affects weight init + FY-loss
# perturbation sampling even without a pruner, so it needs its own seed-1
# run, not reused from the seed-42 baseline), then request/pair/both at
# each threshold. 10 runs total, sequential (Gurobi/multiprocessing-heavy,
# avoid cross-run contention), ~15 min each -> ~2.5h total expected.
set -u

REQUEST_PRUNER_MODEL="outputs/request_pruner_mlp_bi400_ss200/request_pruner_mlp_h32_l1_d0p0_pw1p0_lr0p03/request_pruner_mlp_h32_l1_d0p0_pw1p0_lr0p03_best_val_f3.pt"
SEED=1

run_variant() {
  local variant="$1" threshold="$2" pruner_flags="$3"
  local tag="${variant}"
  if [ -n "$threshold" ]; then tag="${variant}_t${threshold}"; fi
  echo "=== [$(date +%H:%M:%S)] SIL seed=$SEED: $tag ==="
  ./venv/bin/python3 rtv_solver/main.py \
    --mode coaml \
    --input_dir "solutions/li_lim/manifests/" \
    --batch_interval 400 --step_size 200 --max_cardinality 2 \
    --learning_rate 0.0001 --epochs 5 \
    $pruner_flags \
    --seed $SEED \
    --output_dir "outputs/sil_training_bi400_ss200_${tag}_seed${SEED}" \
    > "/tmp/sil_training_bi400_ss200_${tag}_seed${SEED}.log" 2>&1
  echo "=== [$(date +%H:%M:%S)] DONE: $tag seed=$SEED (exit $?) ==="
}

run_variant "baseline" "" "--use_request_pruner False --use_request_graph_pruner False"

for THRESHOLD in 0.4 0.5 0.6; do
  run_variant "request" "$THRESHOLD" "--use_request_pruner True --request_pruner_model_path $REQUEST_PRUNER_MODEL --request_pruner_threshold $THRESHOLD --use_request_graph_pruner False"
  run_variant "pair" "$THRESHOLD" "--use_request_pruner False --use_request_graph_pruner True --request_graph_threshold $THRESHOLD"
  run_variant "both" "$THRESHOLD" "--use_request_pruner True --request_pruner_model_path $REQUEST_PRUNER_MODEL --request_pruner_threshold $THRESHOLD --use_request_graph_pruner True --request_graph_threshold $THRESHOLD"
done

echo "=== ALL 10 SIL seed=$SEED RUNS (baseline + 3 thresholds x 3 variants) DONE ==="
