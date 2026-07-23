#!/bin/bash
# RHO baseline (no pruner), cardinality 2.
set -u

INSTANCES="lc108 lc109 lc207 lc208 lr111 lr112 lr210 lr211 lrc107 lrc108 lrc207 lrc208"

for inst in $INSTANCES; do
  echo ">>> rh/baseline_c2/${inst}"
  time python rtv_solver/main.py \
    --mode offline \
    --input_file "solutions/li_lim/manifests/${inst}.json" \
    --input_dir "solutions/li_lim/manifests/" \
    --imitation_solution_file "solutions/li_lim/manifests/${inst}.json" \
    --use_request_graph_pruner False \
    --max_cardinality 2 --batch_interval 400 --step_size 100 \
    --output_dir "new_tests/cardinality2_compare/bi400_ss100/rh_baseline/${inst}"
done

echo "=== RH BASELINE C2 DONE (12/12) ==="
