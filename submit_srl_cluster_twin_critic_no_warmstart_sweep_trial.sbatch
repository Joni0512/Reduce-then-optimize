#!/bin/bash
#SBATCH --job-name=srl_nows_sw
#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --qos=cm4_serial_long
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=srl_nows_sw_%j.log
#SBATCH --error=srl_nows_sw_%j.err

# One trial of the twin-critic no-warmstart actor_lr x critic_lr x seed wandb sweep - see chat.
# RUN_ID/ACTOR_LR/CRITIC_LR/SEED exported by srl_twin_critic_no_warmstart_sweep_coordinator.py.
# 2026-09-30: moved from serial_std (24h) to serial_long (7 days) - job 5634362 hit the 24h
# TIMEOUT with no result after using serial_std; no-warmstart 30-epoch runs take ~1h/epoch and
# don't reliably finish inside 24h. Trade-off: subject to cm4_serial_long's QOSMaxMemoryPerUser
# cap again (see other _long sbatch scripts).

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"

./venv/bin/python3 -u -m rtv_solver.pipeline.srl_cluster_twin_critic_no_warmstart_sweep_trial "$RUN_ID" "$ACTOR_LR" "$CRITIC_LR" "$SEED"
