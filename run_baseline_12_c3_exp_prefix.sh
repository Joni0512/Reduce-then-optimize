#!/bin/bash
# Cardinality-3 counterpart to the coaml_exponential_prefix c2 baseline batch
# (12 instances, no pruner, exponential_prefix imitation scoring). Each instance
# capped at 30 min wall-clock -- cardinality 3 can blow up combinatorially, see
# run_c3_mixedall_v1final.sh for the same pattern.
set -u
source timeout_helper.sh

INSTANCES="lc108 lc109 lc207 lc208 lr111 lr112 lr210 lr211 lrc107 lrc108 lrc207 lrc208"
LIMIT=1800  # 30 minutes

for inst in $INSTANCES; do
  echo ">>> coaml_exponential_prefix_c3/${inst}"
  time run_with_timeout "$LIMIT" ./venv/bin/python3 rtv_solver/main.py \
    --mode coaml \
    --epochs 1 \
    --input_dir "" \
    --input_file "solutions/li_lim/manifests/${inst}.json" \
    --imitation_solution_file "solutions/li_lim/manifests/${inst}.json" \
    --coaml_solve_mode train \
    --use_request_graph_pruner False \
    --imitation_scoring_rule exponential_prefix \
    --max_cardinality 3 --batch_interval 400 --step_size 100 \
    --seed 42 \
    --output_dir "outputs/coaml_exponential_prefix_c3/${inst}"
done

echo "DONE coaml_exponential_prefix cardinality-3 baseline evaluation"
