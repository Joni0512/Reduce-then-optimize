#!/bin/bash
# 2026-07-17: SIL threshold 0.4 sweep, bi400/ss200 - counterpart to
# run_sil_training_bi400_ss200_all_variants.sh (t=0.5). Skips baseline
# (threshold-independent, already have it from the t=0.5 run).
set -u

REQUEST_PRUNER_MODEL="outputs/request_pruner_mlp_bi400_ss200/request_pruner_mlp_h32_l1_d0p0_pw1p0_lr0p03/request_pruner_mlp_h32_l1_d0p0_pw1p0_lr0p03_best_val_f3.pt"
THRESHOLD=0.4

run_variant() {
  local variant="$1" pruner_flags="$2"
  echo "=== [$(date +%H:%M:%S)] SIL training t=0.4: $variant ==="
  ./venv/bin/python3 rtv_solver/main.py \
    --mode coaml \
    --input_dir "solutions/li_lim/manifests/" \
    --batch_interval 400 --step_size 200 --max_cardinality 2 \
    --learning_rate 0.0001 --epochs 5 \
    $pruner_flags \
    --seed 42 \
    --output_dir "outputs/sil_training_bi400_ss200_${variant}_t0.4" \
    > "/tmp/sil_training_bi400_ss200_${variant}_t0.4.log" 2>&1
  echo "=== [$(date +%H:%M:%S)] DONE: $variant t=0.4 (exit $?) ==="
}

run_variant "request" "--use_request_pruner True --request_pruner_model_path $REQUEST_PRUNER_MODEL --request_pruner_threshold $THRESHOLD --use_request_graph_pruner False"
run_variant "pair" "--use_request_pruner False --use_request_graph_pruner True --request_graph_threshold $THRESHOLD"
run_variant "both" "--use_request_pruner True --request_pruner_model_path $REQUEST_PRUNER_MODEL --request_pruner_threshold $THRESHOLD --use_request_graph_pruner True --request_graph_threshold $THRESHOLD"

echo "=== ALL 3 SIL t=0.4 TRAINING RUNS DONE ==="
