#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "### 1/4 COAML baseline mc=3 ###"
./run_baseline_12_mc3.sh

echo "### 2/4 COAML + pruner sweep mc=3 ###"
./run_coaml_pw5_thresholds_12_mc3.sh

echo "### 3/4 RHO baseline mc=3 ###"
./run_rh_no_learning_baseline_12_mc3.sh

echo "### 4/4 RHO + pruner sweep mc=3 ###"
./run_rho_pw5_thresholds_mc3.sh

echo "ALL 4 CARDINALITY-3 SWEEPS DONE"
