#!/bin/bash
#SBATCH --job-name=srl_rho_outcome_adv_std
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --qos=cm4_serial
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=srl_rho_outcome_adv_std_%j.log
#SBATCH --error=srl_rho_outcome_adv_std_%j.err

# 2026-09-14: serial_std variant (separate QOS/quota from serial_long) - see
# chat. Same job as submit_srl_rho_outcome_advantage_loop.sbatch, tries a
# different queue to avoid waiting behind the two running serial_long jobs.
# If serial_std's actual max walltime is below 48h, sbatch will reject this
# at submit time (same way cm4_tiny rejected 48h) - lower --time if so.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"

./venv/bin/python3 -u -m rtv_solver.pipeline.srl_rho_outcome_advantage_loop 2>&1 | grep --line-buffered -E "^\[rho_outcome_advantage\]|^!!!"
