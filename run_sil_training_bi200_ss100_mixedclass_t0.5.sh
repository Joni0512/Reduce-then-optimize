#!/bin/bash
# 2026-07-20: mixed-class SIL sweep at bi200/ss100, t=0.5 - same setup as
# run_sil_training_bi400_ss200_mixedclass_t0.5.sh (39 training files: 23
# class-1 + 16 class-2, lc204 excluded as a 160-200min/epoch outlier; 12
# validation instances covering both classes).
set -u

REQUEST_PRUNER_MODEL="outputs/request_pruner_mlp_bi200_ss100/request_pruner_mlp_h32_l1_d0p2_pw1p0_lr0p01/request_pruner_mlp_h32_l1_d0p2_pw1p0_lr0p01_best_val_f3.pt"
THRESHOLD=0.5
EXTRA_VAL="lc207,lc208,lr210,lr211,lrc207,lrc208"
EXTRA_TRAIN="lc201,lc202,lc203,lc205,lr201,lr202,lr203,lr204,lr205,lr206,lr207,lrc201,lrc202,lrc203,lrc204,lrc205"

run_variant() {
  local variant="$1" pruner_flags="$2"
  echo "=== [$(date +%H:%M:%S)] SIL bi200/ss100 mixedclass t=0.5: $variant ==="
  ./venv/bin/python3 rtv_solver/main.py \
    --mode coaml \
    --input_dir "solutions/li_lim/manifests/" \
    --batch_interval 200 --step_size 100 --max_cardinality 2 \
    --learning_rate 0.0001 --epochs 5 \
    --extra_validation_files "$EXTRA_VAL" \
    --extra_training_files "$EXTRA_TRAIN" \
    $pruner_flags \
    --seed 42 \
    --output_dir "outputs/sil_training_bi200_ss100_mixedclass_${variant}_t0.5" \
    > "sil_training_bi200_ss100_mixedclass_${variant}_t0.5.log" 2>&1
  echo "=== [$(date +%H:%M:%S)] DONE: $variant (exit $?) ==="
}

run_variant "baseline" "--use_request_pruner False --use_request_graph_pruner False"
run_variant "request" "--use_request_pruner True --request_pruner_model_path $REQUEST_PRUNER_MODEL --request_pruner_threshold $THRESHOLD --use_request_graph_pruner False"
run_variant "pair" "--use_request_pruner False --use_request_graph_pruner True --request_graph_threshold $THRESHOLD"
run_variant "both" "--use_request_pruner True --request_pruner_model_path $REQUEST_PRUNER_MODEL --request_pruner_threshold $THRESHOLD --use_request_graph_pruner True --request_graph_threshold $THRESHOLD"

echo "=== ALL 4 SIL bi200/ss100 mixedclass t=0.5 RUNS DONE ==="
