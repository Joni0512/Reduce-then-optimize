#!/bin/bash
# 2026-10-13: local version of the ablation sweep (see submit_srl_ablation_sweep_cluster.sh): the same 16 runs as N parallel streams on this machine
# (default 2 streams; usage: ./run_srl_ablation_sweep_local.sh [streams]). Logs: ablation_sweep_local_<run>.log in the repo root.
cd "$(dirname "$0")"
ALR=0.0031923396291543868; CLR=0.002
STREAMS=${1:-2}
RUNS=()
for seed in 301 302; do
  for cfg in "sigma0p1 SRL_SIGMA=0.1" "sigma0p3 SRL_SIGMA=0.3" "samples10 SRL_NUM_SAMPLES=10" "samples40 SRL_NUM_SAMPLES=40" \
             "polyak0p001 SRL_POLYAK_TAU=0.001" "polyak0p01 SRL_POLYAK_TAU=0.01" "gamma0p98 SRL_GAMMA=0.98" "gamma0p97 SRL_GAMMA=0.97"; do
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
