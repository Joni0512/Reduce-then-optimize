#!/bin/bash
set -e

INSTANCES=(
  lc101 lc102 lc103 lc104 lc105 lc106 lc107 lc108 lc109 lc110 lc111 lc112
  lc201 lc202 lc203 lc204 lc205 lc206 lc207 lc208
  lr101 lr102 lr103 lr104 lr105 lr106 lr107 lr108 lr109 lr110 lr111 lr112
  lr201 lr202 lr203 lr204 lr205 lr206 lr207 lr208
  lrc101 lrc102 lrc103 lrc104 lrc105 lrc106 lrc107 lrc108 lrc109 lrc110 lrc111 lrc112
  lrc201 lrc202 lrc203 lrc204 lrc205 lrc206 lrc207 lrc208
)

THRESHOLDS=(0.2 0.3 0.5)
MODEL="../outputs/models_v2/rgnn_mixed_c1_pw2_v2/rgnn_mixed_c1_pw2_v2_best_val_f3.pt"

for INSTANCE in "${INSTANCES[@]}"; do
  FILE="solutions/li_lim/manifests/${INSTANCE}.json"

  echo "======================================================"
  echo "RUNNING $INSTANCE | BASELINE no pruner"
  echo "======================================================"

  python main.py \
    --mode coaml \
    --epochs 1 \
    --input_dir "" \
    --input_file "$FILE" \
    --imitation_solution_file "$FILE" \
    --use_request_graph_pruner False \
    --max_cardinality 2 \
    --batch_interval 400 \
    --step_size 100 \
    --output_dir "outputs/eval_mixed_all/${INSTANCE}_coaml_noprune"

  for T in "${THRESHOLDS[@]}"; do
    echo "======================================================"
    echo "RUNNING $INSTANCE | MIXED PRUNER threshold=$T"
    echo "======================================================"

    python main.py \
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
      --output_dir "outputs/eval_mixed_all/${INSTANCE}_coaml_prune_t${T}"
  done
done

echo "DONE mixed model evaluation"
