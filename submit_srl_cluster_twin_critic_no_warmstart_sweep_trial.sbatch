#!/bin/bash
#SBATCH --job-name=srl_nows_sw
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --qos=cm4_serial
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=srl_nows_sw_%j.log
#SBATCH --error=srl_nows_sw_%j.err

# One trial of the twin-critic no-warmstart actor_lr x critic_lr x seed wandb sweep - see chat.
# RUN_ID/ACTOR_LR/CRITIC_LR/SEED exported by srl_twin_critic_no_warmstart_sweep_coordinator.py.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"

./venv/bin/python3 -u -m rtv_solver.pipeline.srl_cluster_twin_critic_no_warmstart_sweep_trial "$RUN_ID" "$ACTOR_LR" "$CRITIC_LR" "$SEED"
