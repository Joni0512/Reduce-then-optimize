#!/bin/bash
#SBATCH --job-name=srl_cm4e20
#SBATCH --clusters=cm4
#SBATCH --partition=cm4_tiny
#SBATCH --qos=cm4_tiny
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=srl_cm4e20_%j.log
#SBATCH --error=srl_cm4e20_%j.err

# cm4 (24 h limit, no per-user CPU/mem QOS limit, 2 jobs on cm4_std / 4 on cm4_tiny) baseline
# protocol for the old per-step SRL architecture - see
# srl_cluster_twin_critic_no_warmstart_pickupslack_cm4_trial.py's docstring. CRITIC_LR, SEED and
# (optionally) EPOCHS exported by the submitter; submit_srl_cm4_ep20_sweep.sh overrides
# --partition/--qos per job. Expected runtime ~19-21 h at EPOCHS=20, so 30 epochs do NOT fit.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"

./venv/bin/python3 -u -m rtv_solver.pipeline.srl_cluster_twin_critic_no_warmstart_pickupslack_cm4_trial "$CRITIC_LR" "$SEED" "${EPOCHS:-20}"
