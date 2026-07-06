#!/bin/bash
set -e

MODEL="outputs/models_v2/rgnn_mixed_all_pw5_v2/rgnn_mixed_all_pw5_v2_best_val_f3.pt"
THRESHOLDS=(0.2 0.3 0.4)

INSTANCES=(
  lc108 lc109 lc207 lc208
  lr111 lr112 lr210 lr211
  lrc107 lrc108 lrc207 lrc208
)

for T in "${THRESHOLDS[@]}"; do
  for INSTANCE in "${INSTANCES[@]}"; do
    FILE="solutions/li_lim/manifests/${INSTANCE}.json"

    echo "===== COAML pw5 | $INSTANCE | threshold=$T ====="

    python3 -m rtv_solver.main \
      --mode coaml \
      --epochs 1 \
      --input_dir "" \
      --input_file "$FILE" \
      --imitation_solution_file "$FILE" \
      --use_request_graph_pruner True \
      --request_graph_model_path "$MODEL" \
      --request_graph_threshold "$T" \
      --max_cardinality 2 \
      --batch_interval 400 \
      --step_size 100 \
      --output_dir "outputs/eval_coaml_pw5_t${T}/${INSTANCE}"
  done
done

echo "DONE COAML pw5 threshold sweep"
