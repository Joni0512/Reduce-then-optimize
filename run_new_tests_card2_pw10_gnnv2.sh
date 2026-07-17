#!/bin/bash
set -e

# 2026-07-07: RH + COAML threshold sweep for the new v2 GNN architecture
# (RequestGraphEdgeGNNv2), best checkpoint by val F3 = pos_weight=10, see
# outputs/models_v2_gnnv2/best_config_by_model_val_f3.csv.
# Baseline (no pruner) runs are NOT repeated here: they don't depend on the
# pruner model at all, and already exist from the pw5 v1 sweep at
# outputs/new_tests/card2_b400_s100_pw5/{rh,coaml}_baseline/ for the same
# instances/card2/batch400/step100 config - reuse those instead of recomputing.
MODEL="outputs/models_v2_gnnv2/rgnn_mixed_c2_pw10_v2/rgnn_mixed_c2_pw10_v2_best_val_f3.pt"
BASE_OUT="outputs/new_tests/card2_b400_s100_pw10_gnnv2"

INSTANCES=(
  lc108 lc109 lc207 lc208
  lr111 lr112 lr210 lr211
  lrc107 lrc108 lrc207 lrc208
)

THRESHOLDS=(0.1 0.2 0.3 0.4 0.5)

echo "=== Running pruner thresholds: RH + COAML (GNN v2, pw10) ==="

for TH in "${THRESHOLDS[@]}"; do
  TH_TAG=$(echo "$TH" | sed 's/\.//')

  for INSTANCE in "${INSTANCES[@]}"; do
    FILE="solutions/li_lim/manifests/${INSTANCE}.json"

    echo "===== RH pruner ${INSTANCE} threshold ${TH} ====="
    python rtv_solver/main.py \
      --mode offline \
      --input_file "$FILE" \
      --imitation_solution_file "$FILE" \
      --use_request_graph_pruner True \
      --request_graph_model_path "$MODEL" \
      --request_graph_threshold "$TH" \
      --max_cardinality 2 \
      --batch_interval 400 \
      --step_size 100 \
      --output_dir "${BASE_OUT}/rh_pruner_t${TH_TAG}/${INSTANCE}"

    echo "===== COAML pruner ${INSTANCE} threshold ${TH} ====="
    python rtv_solver/main.py \
      --mode coaml \
      --input_file "$FILE" \
      --imitation_solution_file "$FILE" \
      --use_request_graph_pruner True \
      --request_graph_model_path "$MODEL" \
      --request_graph_threshold "$TH" \
      --max_cardinality 2 \
      --batch_interval 400 \
      --step_size 100 \
      --output_dir "${BASE_OUT}/coaml_pruner_t${TH_TAG}/${INSTANCE}"
  done
done

echo "DONE new tests card2 b400 s100 pw10 gnnv2"
