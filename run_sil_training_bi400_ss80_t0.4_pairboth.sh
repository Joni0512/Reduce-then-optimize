#!/bin/bash
# 2026-07-18: re-run of pair+both only for bi400/ss80 t0.4 - see
# run_sil_training_bi200_ss40_t0.4_pairboth.sh for why (missing pair-pruner
# GNN checkpoint on cluster, now fixed). request already completed.
set -u

REQUEST_PRUNER_MODEL="outputs/request_pruner_mlp_bi400_ss80/request_pruner_mlp_h64_l3_d0p1_pw3p0_lr0p03/request_pruner_mlp_h64_l3_d0p1_pw3p0_lr0p03_best_val_f3.pt"
THRESHOLD=0.4

run_variant() {
  local variant="$1" pruner_flags="$2"
  echo "=== [$(date +%H:%M:%S)] SIL bi400/ss80 t=0.4: $variant ==="
  ./venv/bin/python3 rtv_solver/main.py \
    --mode coaml \
    --input_dir "solutions/li_lim/manifests/" \
    --batch_interval 400 --step_size 80 --max_cardinality 2 \
    --learning_rate 0.0001 --epochs 5 \
    $pruner_flags \
    --seed 42 \
    --output_dir "outputs/sil_training_bi400_ss80_${variant}_t0.4" \
    > "sil_training_bi400_ss80_${variant}_t0.4.log" 2>&1
  echo "=== [$(date +%H:%M:%S)] DONE: $variant t=0.4 (exit $?) ==="
}

run_variant "pair" "--use_request_pruner False --use_request_graph_pruner True --request_graph_threshold $THRESHOLD"
run_variant "both" "--use_request_pruner True --request_pruner_model_path $REQUEST_PRUNER_MODEL --request_pruner_threshold $THRESHOLD --use_request_graph_pruner True --request_graph_threshold $THRESHOLD"

echo "=== bi400/ss80 pair+both t=0.4 DONE ==="
