#!/bin/bash
#SBATCH --job-name=srl_rho_outcome_adv_12i
#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --qos=cm4_serial_long
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=srl_rho_outcome_adv_12i_%j.log
#SBATCH --error=srl_rho_outcome_adv_12i_%j.err

# 2026-09-15: RHO-outcome-advantage method (same as srl_rho_outcome_advantage_loop.py -
# actor bi200/ss100 with Gaussian perturbation sigma=0.2 vs. live RHO baseline
# bi400/ss100, Option A uniform advantage broadcast, lr=1e-4) run on the
# smaller balanced 12-instance train/test split (see chat, 2026-09-15) instead
# of the full 38-instance split - a quicker test of the method.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"

./venv/bin/python3 -u -m rtv_solver.pipeline.srl_rho_outcome_advantage_loop_12instances
