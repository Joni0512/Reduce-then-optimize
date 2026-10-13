#!/bin/bash
# 2026-10-13: local part of the ablation sweep: only the "samples" ablation (10 and 40 candidates, seeds 301 and 302 = 4 runs, the compute-heavy one); the other
# 12 runs (sigma, Polyak rate, gamma) go to the cluster via submit_srl_ablation_sweep_cluster.sh. N parallel streams (default 2; usage:
# ./run_srl_ablation_sweep_local.sh [streams]). Logs: ablation_sweep_local_<run>.log in the repo root.
cd "$(dirname "$0")"
ALR=0.0031923396291543868; CLR=0.002
STREAMS=${1:-2}
RUNS=()
for seed in 301 302; do
  for cfg in "samples10 SRL_NUM_SAMPLES=10" "samples40 SRL_NUM_SAMPLES=40"; do
    RUNS+=("$cfg $seed")
  done
done
for ((w=0; w<STREAMS; w++)); do
  (
    for ((i=w; i<${#RUNS[@]}; i+=STREAMS)); do
      read -r name envass seed <<< "${RUNS[$i]}"
      env SRL_RUN_ID=abl_${name}_s${seed} $envass ./venv/bin/python3 -u -m rtv_solver.pipeline.srl_buffered_actor_lr_trial $ALR $seed 20 $CLR > ablation_sweep_local_abl_${name}_s${seed}.log 2>&1
    done
  ) &
done
wait
