#!/usr/bin/env bash
set -e
# update config file to rerun the same experiment as stored in the config file
echo "Reproduced experiment"
python main.py \
  --config_file ../outputs/debug/run_20260208_145501_c093fa/config.json \
  --override max_cardinality=3 \
  --override return_depot=False

# add --override to change certain values from the prior experiment and have simple comparisons