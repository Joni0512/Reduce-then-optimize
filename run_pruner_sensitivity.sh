#!/bin/bash
set -e

INSTANCES=("lc108" "lr111")
THRESHOLDS=(0.1 0.3 0.5)
MODEL_PATH="outputs/models_v2/rgnn_mixed_all_pw1_v2/rgnn_mixed_all_pw1_v2_best_val_f3.pt"

for INSTANCE in "${INSTANCES[@]}"; do
  FILE="solutions/li_lim/manifests/${INSTANCE}.json"

  python main.py --mode offline \
    --input_file "$FILE" \
    --max_cardinality 2 --batch_interval 400 --step_size 100 \
    --use_request_graph_pruner False \
    --output_dir "outputs/debug/pruner_sensitivity/${INSTANCE}_rho_noprune"

  python main.py --mode coaml --epochs 1 \
    --input_file "$FILE" \
    --imitation_solution_file "$FILE" \
    --max_cardinality 2 --batch_interval 400 --step_size 100 \
    --use_request_graph_pruner False \
    --output_dir "outputs/debug/pruner_sensitivity/${INSTANCE}_sil_noprune"

  for T in "${THRESHOLDS[@]}"; do
    python main.py --mode offline \
      --input_file "$FILE" \
      --max_cardinality 2 --batch_interval 400 --step_size 100 \
      --use_request_graph_pruner True \
      --request_graph_model_path "$MODEL_PATH" \
      --request_graph_threshold "$T" \
      --output_dir "outputs/debug/pruner_sensitivity/${INSTANCE}_rho_t${T}"

    python main.py --mode coaml --epochs 1 \
      --input_file "$FILE" \
      --imitation_solution_file "$FILE" \
      --max_cardinality 2 --batch_interval 400 --step_size 100 \
      --use_request_graph_pruner True \
      --request_graph_model_path "$MODEL_PATH" \
      --request_graph_threshold "$T" \
      --output_dir "outputs/debug/pruner_sensitivity/${INSTANCE}_sil_t${T}"
  done
done

echo "Fertig. Ergebnisse unter outputs/debug/pruner_sensitivity/*/final/results.json"
