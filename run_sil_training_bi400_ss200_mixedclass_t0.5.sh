#!/bin/bash
# 2026-07-19: SIL sweep at bi400/ss200, t=0.5, with BOTH TRAINING_FILES and
# VALIDATION_FILES extended to mirror the pruners' own mixed-class split
# (MODEL_CONFIGS["mixed_all"]) exactly:
#   - TRAINING_FILES + 17 class-2 train instances (lc201-205, lr201-207, lrc201-205)
#   - VALIDATION_FILES + 6 class-2 test instances (lc207,lc208,lr210,lr211,lrc207,lrc208)
# Motivation: the earlier extval experiment (train=class-1-only, val=both
# classes) confounded pruner-caused damage on class-2 with plain domain shift
# (SIL never having trained on class-2 patterns at all). This run removes
# that confound so class-2 results are attributable to the pruner, matching
# how "generalizes to class 2" is judged for the pruners themselves.
# 39 training files (vs 23 normally) -> each epoch takes longer.
# 2026-07-20: lc204 excluded from EXTRA_TRAIN - empirically found to add
# ~160-200 minutes per epoch on its own (vs ~0.1min for every other file),
# reproduced 3x identically. Every other instance transition takes seconds;
# lc204 alone dominated total runtime. Root cause not fully confirmed (best
# guess: its wider time windows relative to lc201-203 let far more request
# pairs pass the temporal feasibility check, blowing up RTV combination
# generation) but the empirical evidence for exclusion is solid regardless.
set -u

REQUEST_PRUNER_MODEL="outputs/request_pruner_mlp_bi400_ss200/request_pruner_mlp_h32_l1_d0p0_pw1p0_lr0p03/request_pruner_mlp_h32_l1_d0p0_pw1p0_lr0p03_best_val_f3.pt"
THRESHOLD=0.5
EXTRA_VAL="lc207,lc208,lr210,lr211,lrc207,lrc208"
EXTRA_TRAIN="lc201,lc202,lc203,lc205,lr201,lr202,lr203,lr204,lr205,lr206,lr207,lrc201,lrc202,lrc203,lrc204,lrc205"

run_variant() {
  local variant="$1" pruner_flags="$2"
  echo "=== [$(date +%H:%M:%S)] SIL bi400/ss200 mixedclass t=0.5: $variant ==="
  ./venv/bin/python3 rtv_solver/main.py \
    --mode coaml \
    --input_dir "solutions/li_lim/manifests/" \
    --batch_interval 400 --step_size 200 --max_cardinality 2 \
    --learning_rate 0.0001 --epochs 5 \
    --extra_validation_files "$EXTRA_VAL" \
    --extra_training_files "$EXTRA_TRAIN" \
    $pruner_flags \
    --seed 42 \
    --output_dir "outputs/sil_training_bi400_ss200_mixedclass_${variant}_t0.5" \
    > "sil_training_bi400_ss200_mixedclass_${variant}_t0.5.log" 2>&1
  echo "=== [$(date +%H:%M:%S)] DONE: $variant (exit $?) ==="
}

run_variant "baseline" "--use_request_pruner False --use_request_graph_pruner False"
run_variant "request" "--use_request_pruner True --request_pruner_model_path $REQUEST_PRUNER_MODEL --request_pruner_threshold $THRESHOLD --use_request_graph_pruner False"
run_variant "pair" "--use_request_pruner False --use_request_graph_pruner True --request_graph_threshold $THRESHOLD"
run_variant "both" "--use_request_pruner True --request_pruner_model_path $REQUEST_PRUNER_MODEL --request_pruner_threshold $THRESHOLD --use_request_graph_pruner True --request_graph_threshold $THRESHOLD"

echo "=== ALL 4 SIL bi400/ss200 mixedclass t=0.5 RUNS DONE ==="
