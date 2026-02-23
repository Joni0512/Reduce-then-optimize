#!/usr/bin/env bash
set -e
# update config file to rerun the same experiment as stored in the config file
echo "Reproduced experiment"
python main.py \
  --config_file ../outputs/debug/run_20260223_120727_869f27/config.json \
  --override mode=online
# add --override to change certain values from the prior experiment and have simple comparisons (no space between key AND "=" AND value)