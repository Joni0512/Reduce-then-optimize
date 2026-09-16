#!/bin/bash
#SBATCH --job-name=srl_training_trial_std
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --qos=cm4_serial
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=srl_training_trial_std_%j.log
#SBATCH --error=srl_training_trial_std_%j.err

# 2026-09-16: serial_std/cm4_serial variant of submit_srl_cluster_trial.sbatch
# (see chat) - same cluster=serial as the original (so srl_sweep_coordinator.py's
# `squeue -M serial` still finds it), but a DIFFERENT partition/QOS
# (serial_std/cm4_serial instead of serial_long/cm4_serial_long) - a separate
# memory quota pool, so sweep trials submitted here don't wait behind jobs
# already queued on serial_long. 24h time limit (serial_std's confirmed max,
# see CLAUDE.md) instead of serial_long's 48h - should be enough (trials seen
# so far finish in well under 24h).

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"

if [ -z "$RUN_ID" ] || [ -z "$REWARD_MODE" ] || [ -z "$ACTOR_LR" ] || [ -z "$CRITIC_LR" ]; then
    echo "RUN_ID/REWARD_MODE/ACTOR_LR/CRITIC_LR not set - this sbatch is meant to be submitted by srl_sweep_coordinator.py, not run by hand" >&2
    exit 1
fi

./venv/bin/python3 -u -m rtv_solver.pipeline.srl_cluster_trial "$RUN_ID" "$REWARD_MODE" "$ACTOR_LR" "$CRITIC_LR"
