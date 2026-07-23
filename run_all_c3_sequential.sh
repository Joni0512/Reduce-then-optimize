#!/bin/bash
# Runs all cardinality-3 comparisons sequentially: RHO first (baseline, v1_final, v2_final),
# then SIL/COAML (baseline, v1_final, v2_final). One after another, never in parallel.
set -u
cd /Users/joni/Desktop/Masterarbeit/Reduce-then-optimize
source venv/bin/activate

LOG_DIR="new_tests/cardinality3_compare"
mkdir -p "$LOG_DIR"

echo "########## [1/6] RHO baseline ##########"
bash run_rh_baseline_c3.sh > "$LOG_DIR/rh_baseline_c3.log" 2>&1

echo "########## [2/6] RHO + v1_final (mixed_all) ##########"
bash run_rh_mixedall_v1final_c3.sh > "$LOG_DIR/rh_mixedall_v1final_c3.log" 2>&1

echo "########## [3/6] RHO + v2_final (mixed_all) ##########"
bash run_rh_mixedall_v2final_c3.sh > "$LOG_DIR/rh_mixedall_v2final_c3.log" 2>&1

echo "########## [4/6] SIL baseline ##########"
bash run_sil_baseline_c3.sh > "$LOG_DIR/coaml_baseline_c3.log" 2>&1

echo "########## [5/6] SIL + v1_final (mixed_all) ##########"
bash run_sil_mixedall_v1final_c3.sh > "$LOG_DIR/coaml_mixedall_v1final_c3.log" 2>&1

echo "########## [6/6] SIL + v2_final (mixed_all) ##########"
bash run_sil_mixedall_v2final_c3.sh > "$LOG_DIR/coaml_mixedall_v2final_c3.log" 2>&1

echo "=== ALL CARDINALITY-3 RUNS DONE (6/6) ==="
