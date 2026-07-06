#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "### 1/8 COAML baseline mc=3 ###"
./run_baseline_12_mc3.sh

echo "### 2/8 COAML + pruner sweep mc=3 ###"
./run_coaml_pw5_thresholds_12_mc3.sh

echo "### 3/8 RHO baseline mc=3 ###"
./run_rh_no_learning_baseline_12_mc3.sh

echo "### 4/8 RHO + pruner sweep mc=3 ###"
./run_rho_pw5_thresholds_mc3.sh

echo "### 5/8 COAML baseline mc=4 ###"
./run_baseline_12_mc4.sh

echo "### 6/8 COAML + pruner sweep mc=4 ###"
./run_coaml_pw5_thresholds_12_mc4.sh

echo "### 7/8 RHO baseline mc=4 ###"
./run_rh_no_learning_baseline_12_mc4.sh

echo "### 8/8 RHO + pruner sweep mc=4 ###"
./run_rho_pw5_thresholds_mc4.sh

echo "ALL 8 SWEEPS DONE"
