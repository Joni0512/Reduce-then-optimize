#!/bin/bash
# 2026-08-05: class-1-only (standard TRAINING_FILES/VALIDATION_FILES, no
# --extra_training_files/--extra_validation_files) re-test of the Pair
# Pruner (GNN v1_final, outputs/models_v2/rgnn_mixed_c2_pw2_v1_final/) at
# bi400_ss200, thresholds 0.4/0.5/0.6, to verify earlier results. Baseline
# (no pruner) does not depend on threshold, so it is run once and reused
# for comparison against all three pair-pruner thresholds instead of
# being rerun per threshold. --epochs 5 --learning_rate 0.0001 set
# explicitly per thesis Table 3 (main.py defaults of 3/3e-4 do not match).
set -u

PAIR_PRUNER_MODEL="outputs/models_v2/rgnn_mixed_c2_pw2_v1_final/rgnn_mixed_c2_pw2_v1_final_best_val_f3.pt"

echo "=== [$(date +%H:%M:%S)] SIL bi400_ss200 class1 pairpruner v1final: baseline ==="
./venv/bin/python3 rtv_solver/main.py \
  --mode coaml \
  --input_dir "solutions/li_lim/manifests/" \
  --batch_interval 400 --step_size 200 --max_cardinality 2 \
  --learning_rate 0.0001 --epochs 5 \
  --use_request_pruner False --use_request_graph_pruner False \
  --seed 42 \
  --output_dir "outputs/sil_training_bi400_ss200_class1_pairpruner_baseline" \
  > "sil_training_bi400_ss200_class1_pairpruner_baseline.log" 2>&1
echo "=== [$(date +%H:%M:%S)] DONE: baseline (exit $?) ==="

for THRESHOLD in 0.4 0.5 0.6; do
  echo "=== [$(date +%H:%M:%S)] SIL bi400_ss200 class1 pairpruner v1final: pair t=${THRESHOLD} ==="
  ./venv/bin/python3 rtv_solver/main.py \
    --mode coaml \
    --input_dir "solutions/li_lim/manifests/" \
    --batch_interval 400 --step_size 200 --max_cardinality 2 \
    --learning_rate 0.0001 --epochs 5 \
    --use_request_pruner False --use_request_graph_pruner True --request_graph_model_path "$PAIR_PRUNER_MODEL" --request_graph_threshold "$THRESHOLD" \
    --seed 42 \
    --output_dir "outputs/sil_training_bi400_ss200_class1_pairpruner_pair_t${THRESHOLD}" \
    > "sil_training_bi400_ss200_class1_pairpruner_pair_t${THRESHOLD}.log" 2>&1
  echo "=== [$(date +%H:%M:%S)] DONE: pair t=${THRESHOLD} (exit $?) ==="
done

echo "=== ALL SIL bi400_ss200 class1 pairpruner v1final RUNS DONE (4/4) ==="
