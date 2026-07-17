#!/bin/bash
# 2026-07-17: real SIL training (mode="train", 5 epochs, backprop) for all 4
# pruner variants at bi400/ss200, threshold 0.5 for both pruners (uniform,
# matching the RHO/OPT sweeps). Sequential, not parallel - each run is
# multiprocessing/Gurobi-heavy internally, avoid cross-run contention.
set -u

REQUEST_PRUNER_MODEL="outputs/request_pruner_mlp_bi400_ss200/request_pruner_mlp_h32_l1_d0p0_pw1p0_lr0p03/request_pruner_mlp_h32_l1_d0p0_pw1p0_lr0p03_best_val_f3.pt"
THRESHOLD=0.5

run_variant() {
  local variant="$1" pruner_flags="$2"
  echo "=== [$(date +%H:%M:%S)] SIL training: $variant ==="
  ./venv/bin/python3 rtv_solver/main.py \
    --mode coaml \
    --input_dir "solutions/li_lim/manifests/" \
    --batch_interval 400 --step_size 200 --max_cardinality 2 \
    --learning_rate 0.0001 --epochs 5 \
    $pruner_flags \
    --seed 42 \
    --output_dir "outputs/sil_training_bi400_ss200_${variant}_t0.5" \
    > "/tmp/sil_training_bi400_ss200_${variant}_t0.5.log" 2>&1
  echo "=== [$(date +%H:%M:%S)] DONE: $variant (exit $?) ==="
}

run_variant "baseline" "--use_request_pruner False --use_request_graph_pruner False"
run_variant "request" "--use_request_pruner True --request_pruner_model_path $REQUEST_PRUNER_MODEL --request_pruner_threshold $THRESHOLD --use_request_graph_pruner False"
run_variant "pair" "--use_request_pruner False --use_request_graph_pruner True --request_graph_threshold $THRESHOLD"
run_variant "both" "--use_request_pruner True --request_pruner_model_path $REQUEST_PRUNER_MODEL --request_pruner_threshold $THRESHOLD --use_request_graph_pruner True --request_graph_threshold $THRESHOLD"

echo "=== ALL 4 SIL TRAINING RUNS DONE ==="
