#!/bin/bash
#SBATCH --job-name=bc_vs_rho_bi800
#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --qos=cm4_serial_long
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=srl_bc_vs_rho_bi800_%j.log
#SBATCH --error=srl_bc_vs_rho_bi800_%j.err

# 2026-09-17: Behavior Cloning (Option 1) against a live RHO baseline at
# bi800/ss100 (widened further from bi400/ss100), actor stays at bi200/ss100
# - see rtv_solver/pipeline/srl_behavior_cloning_vs_rho_loop_bi800.py's
# module docstring and chat, 2026-09-17.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"

./venv/bin/python3 -u -m rtv_solver.pipeline.srl_behavior_cloning_vs_rho_loop_bi800
