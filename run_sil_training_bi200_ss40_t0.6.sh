#!/bin/bash
# 2026-07-18: SIL threshold 0.6 sweep, bi200/ss40 - this is the config that
# showed the most extreme fallback/direct-miss damage at t=0.5 (45.7%
# fallback at t=0, 62.9% direct miss vs RHO baseline). Checking whether t=0.6
# makes it even more extreme (mirrors run_sil_training_bi400_ss200_t0.6.sh).
# Skips baseline (threshold-independent, already have it from the t=0.5 run).
set -u

REQUEST_PRUNER_MODEL="outputs/request_pruner_mlp_bi200_ss40/request_pruner_mlp_h64_l1_d0p1_pw1p0_lr0p01/request_pruner_mlp_h64_l1_d0p1_pw1p0_lr0p01_best_val_f3.pt"
THRESHOLD=0.6

run_variant() {
  local variant="$1" pruner_flags="$2"
  echo "=== [$(date +%H:%M:%S)] SIL bi200/ss40 t=0.6: $variant ==="
  ./venv/bin/python3 rtv_solver/main.py \
    --mode coaml \
    --input_dir "solutions/li_lim/manifests/" \
    --batch_interval 200 --step_size 40 --max_cardinality 2 \
    --learning_rate 0.0001 --epochs 5 \
    $pruner_flags \
    --seed 42 \
    --output_dir "outputs/sil_training_bi200_ss40_${variant}_t0.6" \
    > "sil_training_bi200_ss40_${variant}_t0.6.log" 2>&1
  echo "=== [$(date +%H:%M:%S)] DONE: $variant t=0.6 (exit $?) ==="
}

run_variant "request" "--use_request_pruner True --request_pruner_model_path $REQUEST_PRUNER_MODEL --request_pruner_threshold $THRESHOLD --use_request_graph_pruner False"
run_variant "pair" "--use_request_pruner False --use_request_graph_pruner True --request_graph_threshold $THRESHOLD"
run_variant "both" "--use_request_pruner True --request_pruner_model_path $REQUEST_PRUNER_MODEL --request_pruner_threshold $THRESHOLD --use_request_graph_pruner True --request_graph_threshold $THRESHOLD"

echo "=== ALL 3 SIL bi200/ss40 t=0.6 TRAINING RUNS DONE ==="
