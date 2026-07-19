#!/bin/bash
# 2026-07-19: SIL sweep at bi400/ss200, t=0.4, with BOTH TRAINING_FILES and
# VALIDATION_FILES extended to mirror the pruners' own mixed-class split
# (see run_sil_training_bi400_ss200_mixedclass_t0.5.sh for the full
# rationale). This is the t=0.4 counterpart, meant to run on the cluster
# in parallel with the local t=0.5 mixedclass run.
set -u

REQUEST_PRUNER_MODEL="outputs/request_pruner_mlp_bi400_ss200/request_pruner_mlp_h32_l1_d0p0_pw1p0_lr0p03/request_pruner_mlp_h32_l1_d0p0_pw1p0_lr0p03_best_val_f3.pt"
THRESHOLD=0.4
EXTRA_VAL="lc207,lc208,lr210,lr211,lrc207,lrc208"
EXTRA_TRAIN="lc201,lc202,lc203,lc204,lc205,lr201,lr202,lr203,lr204,lr205,lr206,lr207,lrc201,lrc202,lrc203,lrc204,lrc205"

run_variant() {
  local variant="$1" pruner_flags="$2"
  echo "=== [$(date +%H:%M:%S)] SIL bi400/ss200 mixedclass t=0.4: $variant ==="
  ./venv/bin/python3 rtv_solver/main.py \
    --mode coaml \
    --input_dir "solutions/li_lim/manifests/" \
    --batch_interval 400 --step_size 200 --max_cardinality 2 \
    --learning_rate 0.0001 --epochs 5 \
    --extra_validation_files "$EXTRA_VAL" \
    --extra_training_files "$EXTRA_TRAIN" \
    $pruner_flags \
    --seed 42 \
    --output_dir "outputs/sil_training_bi400_ss200_mixedclass_${variant}_t0.4" \
    > "sil_training_bi400_ss200_mixedclass_${variant}_t0.4.log" 2>&1
  echo "=== [$(date +%H:%M:%S)] DONE: $variant (exit $?) ==="
}

run_variant "baseline" "--use_request_pruner False --use_request_graph_pruner False"
run_variant "request" "--use_request_pruner True --request_pruner_model_path $REQUEST_PRUNER_MODEL --request_pruner_threshold $THRESHOLD --use_request_graph_pruner False"
run_variant "pair" "--use_request_pruner False --use_request_graph_pruner True --request_graph_threshold $THRESHOLD"
run_variant "both" "--use_request_pruner True --request_pruner_model_path $REQUEST_PRUNER_MODEL --request_pruner_threshold $THRESHOLD --use_request_graph_pruner True --request_graph_threshold $THRESHOLD"

echo "=== ALL 4 SIL bi400/ss200 mixedclass t=0.4 RUNS DONE ==="
