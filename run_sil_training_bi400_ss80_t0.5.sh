#!/bin/bash
# 2026-07-17: SIL sweep at bi400/ss80 (ratio 5:1 - horizon fixed at 400,
# same as the bi400/ss200 runs, but step size shrunk from 200 to 80 to
# isolate whether the horizon:step RATIO matters, not just absolute size).
# Threshold 0.5, seed 42, 4 variants sequentially.
set -u

REQUEST_PRUNER_MODEL_DIR="outputs/request_pruner_mlp_bi400_ss80"
REQUEST_PRUNER_MODEL=$(find "$REQUEST_PRUNER_MODEL_DIR" -iname "*_best_val_f3.pt" | grep -v sweep | head -1)
THRESHOLD=0.5
SEED=42

if [ -z "$REQUEST_PRUNER_MODEL" ]; then
  echo "!!! No trained request pruner checkpoint found under $REQUEST_PRUNER_MODEL_DIR - aborting."
  exit 1
fi
echo "Using request pruner checkpoint: $REQUEST_PRUNER_MODEL"

run_variant() {
  local variant="$1" pruner_flags="$2"
  echo "=== [$(date +%H:%M:%S)] SIL bi400/ss80 t=0.5: $variant ==="
  ./venv/bin/python3 rtv_solver/main.py \
    --mode coaml \
    --input_dir "solutions/li_lim/manifests/" \
    --batch_interval 400 --step_size 80 --max_cardinality 2 \
    --learning_rate 0.0001 --epochs 5 \
    $pruner_flags \
    --seed $SEED \
    --output_dir "outputs/sil_training_bi400_ss80_${variant}_t0.5" \
    > "/tmp/sil_training_bi400_ss80_${variant}_t0.5.log" 2>&1
  echo "=== [$(date +%H:%M:%S)] DONE: $variant (exit $?) ==="
}

run_variant "baseline" "--use_request_pruner False --use_request_graph_pruner False"
run_variant "request" "--use_request_pruner True --request_pruner_model_path $REQUEST_PRUNER_MODEL --request_pruner_threshold $THRESHOLD --use_request_graph_pruner False"
run_variant "pair" "--use_request_pruner False --use_request_graph_pruner True --request_graph_threshold $THRESHOLD"
run_variant "both" "--use_request_pruner True --request_pruner_model_path $REQUEST_PRUNER_MODEL --request_pruner_threshold $THRESHOLD --use_request_graph_pruner True --request_graph_threshold $THRESHOLD"

echo "=== ALL 4 SIL bi400/ss80 t=0.5 RUNS DONE ==="
