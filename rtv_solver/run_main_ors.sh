#!/usr/bin/env bash
set -e
# update config file to rerun the same experiment as stored in the config file
echo "Reproduced experiment"
python main_ors.py \
  --config_file ../outputs/debug/run_20260208_131743_9c1ad3/config.json