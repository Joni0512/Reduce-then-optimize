#!/bin/bash
#SBATCH --job-name=srl_rho_outcome_adv_cm4
#SBATCH --clusters=cm4
#SBATCH --partition=cm4_tiny
#SBATCH --qos=cm4_tiny
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=srl_rho_outcome_adv_cm4_%j.log
#SBATCH --error=srl_rho_outcome_adv_cm4_%j.err

# 2026-09-14: same as submit_srl_rho_outcome_advantage_loop.sbatch, but on
# the separate cm4/cm4_tiny queue instead of serial - see chat. Used to run
# in parallel without waiting behind the two already-running srl_training_loop
# sweep trials on the serial cluster.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"

./venv/bin/python3 -u -m rtv_solver.pipeline.srl_rho_outcome_advantage_loop
