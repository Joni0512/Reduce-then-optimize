#!/usr/bin/env bash
set -e
# update config file to rerun the same experiment as stored in the config file
echo "Reproduced experiment"
python main.py \
  --config_file ../outputs/storage/comp_v1/run_20260223_164646_optimal?/config.json \
  --override mode=offline \
  --override max_cardinality=6 \
  --override largest_tsp=12 \
  --override debug=True \
  --override rtv_timeout=300 \
  --override share_cost_factor=10 \
  --override step_size=500 \
  --override batch_interval=1500
  # add --override to change certain values from the prior experiment and have simple comparisons (no space between key AND "=" AND value) and add \ at the end of each line that is not the last