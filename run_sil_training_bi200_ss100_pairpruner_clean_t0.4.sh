#!/bin/bash
# 2026-08-03: clean re-test of the Pair Pruner (GNN v1, "v1_final" checkpoint,
# outputs/models_v2/rgnn_mixed_c2_pw2_v1_final/), bi200/ss100, t=0.4.
# Same mixed-class SIL setup as run_sil_training_bi200_ss100_mixedclass_t0.4.sh
# (39 training files: 23 class-1 + 16 class-2, lc204 excluded as a
# 160-200min/epoch outlier; 12 validation instances covering both classes) -
# reused here so the Pair Pruner is judged against a SIL model that was
# itself trained on the same class mix the pruner was trained on, avoiding
# the domain-shift confound found earlier this session. --epochs 5
# --learning_rate 0.0001 set explicitly per thesis Table 3 (main.py's
# argparse defaults of 3/3e-4 do not match).
set -u

PAIR_PRUNER_MODEL="outputs/models_v2/rgnn_mixed_c2_pw2_v1_final/rgnn_mixed_c2_pw2_v1_final_best_val_f3.pt"
THRESHOLD=0.4
EXTRA_VAL="lc207,lc208,lr210,lr211,lrc207,lrc208"
EXTRA_TRAIN="lc201,lc202,lc203,lc205,lr201,lr202,lr203,lr204,lr205,lr206,lr207,lrc201,lrc202,lrc203,lrc204,lrc205"

echo "=== [$(date +%H:%M:%S)] SIL bi200/ss100 pairpruner-clean t=0.4: baseline ==="
./venv/bin/python3 rtv_solver/main.py \
  --mode coaml \
  --input_dir "solutions/li_lim/manifests/" \
  --batch_interval 200 --step_size 100 --max_cardinality 2 \
  --learning_rate 0.0001 --epochs 5 \
  --extra_validation_files "$EXTRA_VAL" \
  --extra_training_files "$EXTRA_TRAIN" \
  --use_request_pruner False --use_request_graph_pruner False \
  --seed 42 \
  --output_dir "outputs/sil_training_bi200_ss100_pairpruner_clean_baseline_t0.4" \
  > "sil_training_bi200_ss100_pairpruner_clean_baseline_t0.4.log" 2>&1
echo "=== [$(date +%H:%M:%S)] DONE: baseline (exit $?) ==="

echo "=== [$(date +%H:%M:%S)] SIL bi200/ss100 pairpruner-clean t=0.4: pair ==="
./venv/bin/python3 rtv_solver/main.py \
  --mode coaml \
  --input_dir "solutions/li_lim/manifests/" \
  --batch_interval 200 --step_size 100 --max_cardinality 2 \
  --learning_rate 0.0001 --epochs 5 \
  --extra_validation_files "$EXTRA_VAL" \
  --extra_training_files "$EXTRA_TRAIN" \
  --use_request_pruner False --use_request_graph_pruner True --request_graph_model_path "$PAIR_PRUNER_MODEL" --request_graph_threshold "$THRESHOLD" \
  --seed 42 \
  --output_dir "outputs/sil_training_bi200_ss100_pairpruner_clean_pair_t0.4" \
  > "sil_training_bi200_ss100_pairpruner_clean_pair_t0.4.log" 2>&1
echo "=== [$(date +%H:%M:%S)] DONE: pair (exit $?) ==="

echo "=== ALL SIL bi200/ss100 pairpruner-clean t=0.4 RUNS DONE ==="
