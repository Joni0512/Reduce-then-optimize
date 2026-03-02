#!/usr/bin/env bash
set -e
# update config file to rerun the same experiment as stored in the config file
INPUT_FILE="test_nc/test_10r_1v_repeat8.json" # stored in rtv_solver/inputs/
python main.py \
  --input_file $INPUT_FILE \
  --mode optimal_solution \
  --no-debug
echo "Run complete"