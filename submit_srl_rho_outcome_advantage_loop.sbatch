#!/bin/bash
#SBATCH --job-name=srl_rho_outcome_adv
#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --qos=cm4_serial_long
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=srl_rho_outcome_adv_%j.log
#SBATCH --error=srl_rho_outcome_adv_%j.err

# 2026-09-14: first full 38-instance/30-epoch run of the RHO-outcome-advantage
# method (see chat/srl deck) - actor at bi200/ss100 vs. a live RHO baseline
# (COAMLPipeline mode="offline") at bi400/ss100, Option A uniform advantage
# broadcast, lr=1e-4 (see srl_rho_outcome_advantage.py's single-instance
# smoke test - lr=1e-2 over-corrected and destabilized after a few epochs).
# ~2400 solver calls total, expect this to run considerably longer than the
# existing ~1200-call srl_training_loop.py sweep trials - 48h time limit as
# a first estimate, adjust if it's not enough.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"

./venv/bin/python3 -u -m rtv_solver.pipeline.srl_rho_outcome_advantage_loop 2>&1 | grep --line-buffered -E "^\[rho_outcome_advantage\]|^!!!"
