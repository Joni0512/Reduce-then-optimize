#!/usr/bin/env bash
set -e
# update config file to rerun the same experiment as stored in the config file

### EXPERIMENT PARAMETERS (from run_main.sh)

INPUT_DIR="solutions/li_lim/manifests"
OUTPUT_DIR_NAME="experiment_fresh"
MAX_THREAD_CNT=16
MAX_CARDINALITY=4
STEP_SIZE=100
BATCH_INTERVAL=200
SEED=42
MODE="coaml"
EPOCHS=5
LEARNING_RATE=1e-4

python rtv_solver/main.py \
  --input_dir $INPUT_DIR \
  --output_dir $OUTPUT_DIR_NAME \
  --max_thread_cnt $MAX_THREAD_CNT \
  --max_cardinality $MAX_CARDINALITY \
  --step_size $STEP_SIZE \
  --batch_interval $BATCH_INTERVAL \
  --seed $SEED \
  --mode $MODE \
  --epochs $EPOCHS \
  --learning_rate $LEARNING_RATE

echo "Run complete"