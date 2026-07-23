#!/bin/bash
# 2026-07-18: re-run of pair+both only for bi200/ss40 t0.4 - the first
# cluster attempt crashed both variants immediately with FileNotFoundError
# on the pair-pruner GNN checkpoint (outputs/models_v2_gnnv2/rgnn_mixed_c2_pw10_v2/...),
# which was never uploaded to the cluster (only the 3 config-specific
# request-pruner MLP checkpoints were). request already completed
# successfully and is not re-run here.
set -u

REQUEST_PRUNER_MODEL="outputs/request_pruner_mlp_bi200_ss40/request_pruner_mlp_h64_l1_d0p1_pw1p0_lr0p01/request_pruner_mlp_h64_l1_d0p1_pw1p0_lr0p01_best_val_f3.pt"
THRESHOLD=0.4

run_variant() {
  local variant="$1" pruner_flags="$2"
  echo "=== [$(date +%H:%M:%S)] SIL bi200/ss40 t=0.4: $variant ==="
  ./venv/bin/python3 rtv_solver/main.py \
    --mode coaml \
    --input_dir "solutions/li_lim/manifests/" \
    --batch_interval 200 --step_size 40 --max_cardinality 2 \
    --learning_rate 0.0001 --epochs 5 \
    $pruner_flags \
    --seed 42 \
    --output_dir "outputs/sil_training_bi200_ss40_${variant}_t0.4" \
    > "sil_training_bi200_ss40_${variant}_t0.4.log" 2>&1
  echo "=== [$(date +%H:%M:%S)] DONE: $variant t=0.4 (exit $?) ==="
}

run_variant "pair" "--use_request_pruner False --use_request_graph_pruner True --request_graph_threshold $THRESHOLD"
run_variant "both" "--use_request_pruner True --request_pruner_model_path $REQUEST_PRUNER_MODEL --request_pruner_threshold $THRESHOLD --use_request_graph_pruner True --request_graph_threshold $THRESHOLD"

echo "=== bi200/ss40 pair+both t=0.4 DONE ==="
