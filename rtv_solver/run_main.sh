#!/usr/bin/env bash
set -e
# update config file to rerun the same experiment as stored in the config file
INPUT_FILE="wilson_nc_initial.pkl" # stored in rtv_solver/inputs/
python main.py \
  --max_cardinality 3 \
  --mode online \
  --batch_interval 3600 \
  --step_size 1200
echo "Run complete"