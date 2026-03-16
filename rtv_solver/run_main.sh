#!/usr/bin/env bash
set -e
# update config file to rerun the same experiment as stored in the config file
OUTPUT_DIR="experiments"
MAX_THREAD_CNT=128
MAX_CARDINALITY=3
STEP_SIZE=100
BATCH_INTERVAL=400
SEED=42
MODE=offline
EPOCHS=10
LEARNING_RATE=5e-3
INPUT_FILE="solutions/li_lim/manifests/lc101.json"
IMITATION_SOLUTION_FILE="solutions/li_lim/manifests/lc101.json"

python main.py \
  --output_dir "$OUTPUT_DIR" \
  --max_thread_cnt $MAX_THREAD_CNT \
  --max_cardinality $MAX_CARDINALITY \
  --step_size $STEP_SIZE \
  --batch_interval $BATCH_INTERVAL \
  --seed $SEED \
  --mode $MODE \
  --epochs $EPOCHS \
  --learning_rate $LEARNING_RATE \
  --input_file "$INPUT_FILE" \
  --imitation_solution_file "$IMITATION_SOLUTION_FILE"
echo "Run complete"