#!/bin/bash
# 2026-10-07: local agent of the buffered-actor wandb sweep (see sweep_srl_buffered.yaml). First call creates
# the sweep (needs `wandb login` done once) and stores its ID in sweep_srl_buffered_id.txt - give that ID to the
# cluster coordinators too (run_srl_buffered_sweep_cluster_coordinator.sh). Re-run to resume after an interruption;
# the 50-run budget (run_cap) is enforced by wandb's server, not by this script.
# Usage: ./run_srl_buffered_sweep_local.sh
set -e
cd "$(dirname "$0")"
ID_FILE="sweep_srl_buffered_id.txt"
if [ ! -f "$ID_FILE" ]; then
    ./venv/bin/wandb sweep sweep_srl_buffered.yaml 2>&1 | tee /tmp/sweep_srl_buffered_create.log
    SWEEP_ID=$(grep -oE "[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+/[a-z0-9]+$" /tmp/sweep_srl_buffered_create.log | tail -1)
    [ -n "$SWEEP_ID" ] || { echo "could not parse sweep ID, see /tmp/sweep_srl_buffered_create.log" >&2; exit 1; }
    echo "$SWEEP_ID" > "$ID_FILE"
fi
SWEEP_ID=$(cat "$ID_FILE")
echo "sweep: $SWEEP_ID"
./venv/bin/wandb agent "$SWEEP_ID" 2>&1 | tee -a srl_buffered_sweep_local.log
