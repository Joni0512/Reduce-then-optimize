#!/bin/bash
# 2026-10-07: cluster coordinator of the buffered-actor wandb sweep - run on a LOGIN node (needs internet), not
# via sbatch. One call = one concurrent cluster job slot; start several (e.g. 4 for cm4_tiny) in parallel.
# Usage: nohup ./run_srl_buffered_sweep_cluster_coordinator.sh <sweep_id> <cm4|serial> [count=50] > coord_1.log 2>&1 &
# Restarts the coordinator only if it exits with an error; a normal exit means the sweep has no runs left.
set -u
cd "$(dirname "$0")"
SWEEP_ID="$1"; CLUSTER="${2:-cm4}"; COUNT="${3:-50}"
for attempt in 1 2 3 4 5; do
    ./venv/bin/python3 -u -m rtv_solver.pipeline.srl_buffered_sweep_coordinator "$SWEEP_ID" "$COUNT" "$CLUSTER" && exit 0
    echo "$(date): coordinator exited with an error (attempt $attempt/5) - restarting in 60s"
    sleep 60
done
