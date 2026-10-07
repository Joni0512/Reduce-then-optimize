#!/bin/bash
#SBATCH --job-name=srl_buf
#SBATCH --clusters=cm4
#SBATCH --partition=cm4_tiny
#SBATCH --qos=cm4_tiny
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --output=srl_buf_%j.log
#SBATCH --error=srl_buf_%j.err

# 2026-10-06: actor_lr scan of the "collect, then update" actor scheme - see
# srl_buffered_actor_lr_trial.py's docstring. One run per job: ACTOR_LR (required), SEED (default 42)
# and EPOCHS (default 20) exported by the submitter. cm4_tiny needs >= 17 CPUs per job (see
# submit_srl_cluster_twin_critic_no_warmstart_pickupslack_cm4_trial.sbatch); 64 G is a guess based on
# the old architecture's <= 18 G MaxRSS plus the new per-window snapshot buffer (not measured yet).
# Submit: sbatch --export=ALL,ACTOR_LR=0.0064,SEED=42,EPOCHS=20 submit_srl_buffered_actor_lr_trial.sbatch

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"

./venv/bin/python3 -u -m rtv_solver.pipeline.srl_buffered_actor_lr_trial "$ACTOR_LR" "${SEED:-42}" "${EPOCHS:-20}" ${CRITIC_LR:+"$CRITIC_LR"}  # 2026-10-07: optional CRITIC_LR (wandb sweep)
