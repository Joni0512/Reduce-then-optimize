#!/bin/bash
# 2026-07-17: SIL (mode="train"+eval) sweep at bi200/ss100 (more frequent
# re-optimization than the bi400/ss200 runs so far), threshold 0.5, seed 42.
# 4 variants (baseline/request/pair/both) sequentially.
set -u

REQUEST_PRUNER_MODEL="outputs/request_pruner_mlp_bi200_ss100/request_pruner_mlp_h32_l1_d0p2_pw1p0_lr0p01/request_pruner_mlp_h32_l1_d0p2_pw1p0_lr0p01_best_val_f3.pt"
THRESHOLD=0.5
SEED=42

run_variant() {
  local variant="$1" pruner_flags="$2"
  echo "=== [$(date +%H:%M:%S)] SIL bi200/ss100 t=0.5: $variant ==="
  ./venv/bin/python3 rtv_solver/main.py \
    --mode coaml \
    --input_dir "solutions/li_lim/manifests/" \
    --batch_interval 200 --step_size 100 --max_cardinality 2 \
    --learning_rate 0.0001 --epochs 5 \
    $pruner_flags \
    --seed $SEED \
    --output_dir "outputs/sil_training_bi200_ss100_${variant}_t0.5" \
    > "/tmp/sil_training_bi200_ss100_${variant}_t0.5.log" 2>&1
  echo "=== [$(date +%H:%M:%S)] DONE: $variant (exit $?) ==="
}

run_variant "baseline" "--use_request_pruner False --use_request_graph_pruner False"
run_variant "request" "--use_request_pruner True --request_pruner_model_path $REQUEST_PRUNER_MODEL --request_pruner_threshold $THRESHOLD --use_request_graph_pruner False"
run_variant "pair" "--use_request_pruner False --use_request_graph_pruner True --request_graph_threshold $THRESHOLD"
run_variant "both" "--use_request_pruner True --request_pruner_model_path $REQUEST_PRUNER_MODEL --request_pruner_threshold $THRESHOLD --use_request_graph_pruner True --request_graph_threshold $THRESHOLD"

echo "=== ALL 4 SIL bi200/ss100 t=0.5 RUNS DONE ==="
