#!/bin/bash
#SBATCH --job-name=srl_training_trial
#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --qos=cm4_serial_long
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=srl_training_trial_%j.log
#SBATCH --error=srl_training_trial_%j.err

# 2026-09-13: one trial of the SIL/SRL(local)/SRL(local_positive) actor_lr x
# critic_lr wandb sweep, submitted by srl_sweep_coordinator.py (running on
# the login node, which has internet - this job itself never talks to
# wandb) - see chat/srl deck. RUN_ID/REWARD_MODE/ACTOR_LR/CRITIC_LR are
# passed via --export by the coordinator.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"

if [ -z "$RUN_ID" ] || [ -z "$REWARD_MODE" ] || [ -z "$ACTOR_LR" ] || [ -z "$CRITIC_LR" ]; then
    echo "RUN_ID/REWARD_MODE/ACTOR_LR/CRITIC_LR not set - this sbatch is meant to be submitted by srl_sweep_coordinator.py, not run by hand" >&2
    exit 1
fi

./venv/bin/python3 -u -m rtv_solver.pipeline.srl_cluster_trial "$RUN_ID" "$REWARD_MODE" "$ACTOR_LR" "$CRITIC_LR"
