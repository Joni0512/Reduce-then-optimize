#!/bin/bash
# 2026-07-18: SIL sweep at bi400/ss200, max_cardinality=3 (up from the
# cardinality=2 used everywhere else so far). Cardinality 3 blows up RTV
# combinatorics significantly (see thesis Section 6.2: ~90min-2h/epoch at
# cardinality 3 vs ~20-30min at cardinality 2, for the 50-10 config) - this
# is why it's meant to run on the cluster, not locally. Threshold 0.5,
# seed 42, 4 variants (baseline/request/pair/both) sequentially.
#
# Note: the request pruner's own decision logic (13 pre-solve features,
# no cardinality dependence) and the pair pruner's GNN (trained on the
# static optimal solution, also cardinality-independent) both reuse the
# EXISTING bi400/ss200 checkpoints unchanged - cardinality only affects
# trip generation downstream of pruning, not the pruners themselves.
set -u

REQUEST_PRUNER_MODEL="outputs/request_pruner_mlp_bi400_ss200/request_pruner_mlp_h32_l1_d0p0_pw1p0_lr0p03/request_pruner_mlp_h32_l1_d0p0_pw1p0_lr0p03_best_val_f3.pt"
THRESHOLD=0.5
SEED=42
CARDINALITY=3

run_variant() {
  local variant="$1" pruner_flags="$2"
  echo "=== [$(date +%H:%M:%S)] SIL bi400/ss200 mc3 t=0.5: $variant ==="
  ./venv/bin/python3 rtv_solver/main.py \
    --mode coaml \
    --input_dir "solutions/li_lim/manifests/" \
    --batch_interval 400 --step_size 200 --max_cardinality $CARDINALITY \
    --learning_rate 0.0001 --epochs 5 \
    $pruner_flags \
    --seed $SEED \
    --output_dir "outputs/sil_training_bi400_ss200_mc3_${variant}_t0.5" \
    > "sil_training_bi400_ss200_mc3_${variant}_t0.5.log" 2>&1
  echo "=== [$(date +%H:%M:%S)] DONE: $variant (exit $?) ==="
}

run_variant "baseline" "--use_request_pruner False --use_request_graph_pruner False"
run_variant "request" "--use_request_pruner True --request_pruner_model_path $REQUEST_PRUNER_MODEL --request_pruner_threshold $THRESHOLD --use_request_graph_pruner False"
run_variant "pair" "--use_request_pruner False --use_request_graph_pruner True --request_graph_threshold $THRESHOLD"
run_variant "both" "--use_request_pruner True --request_pruner_model_path $REQUEST_PRUNER_MODEL --request_pruner_threshold $THRESHOLD --use_request_graph_pruner True --request_graph_threshold $THRESHOLD"

echo "=== ALL 4 SIL bi400/ss200 mc3 t=0.5 RUNS DONE ==="
