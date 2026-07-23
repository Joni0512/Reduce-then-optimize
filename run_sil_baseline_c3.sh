#!/bin/bash
# SIL (Structured Imitation Learning / COAML) baseline (no pruner), cardinality 3.
# Part of the 4-way cardinality-3 comparison (RHO no-learning / RHO+GNN / SIL no-pruner / SIL+GNN).
set -u

INSTANCES="lc108 lc109 lc207 lc208 lr111 lr112 lr210 lr211 lrc107 lrc108 lrc207 lrc208"

for inst in $INSTANCES; do
  echo ">>> coaml/baseline_c3/${inst}"
  time python rtv_solver/main.py \
    --mode coaml --input_dir "" \
    --input_file "solutions/li_lim/manifests/${inst}.json" \
    --imitation_solution_file "solutions/li_lim/manifests/${inst}.json" \
    --use_request_graph_pruner False \
    --max_cardinality 3 --batch_interval 400 --step_size 100 \
    --output_dir "new_tests/cardinality3_compare/bi400_ss100/coaml_baseline/${inst}"
done

echo "=== COAML BASELINE C3 DONE (12/12) ==="
