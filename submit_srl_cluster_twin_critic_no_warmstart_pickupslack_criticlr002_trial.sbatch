#!/bin/bash
#SBATCH --job-name=srl_nows_ps2
#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --qos=cm4_serial_long
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=srl_nows_ps2_%j.log
#SBATCH --error=srl_nows_ps2_%j.err

# No-warmstart twin-critic, seed=42, 30 epochs, pickup_slack=True, critic_lr=0.002 - see chat
# and srl_cluster_twin_critic_no_warmstart_pickupslack_criticlr002_trial.py's own docstring.
# serial_long chosen directly (not serial_std) - no-warmstart 30-epoch runs take ~1h/epoch and
# a serial_std job (5634362) already hit the 24h TIMEOUT with no result.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"

./venv/bin/python3 -u -m rtv_solver.pipeline.srl_cluster_twin_critic_no_warmstart_pickupslack_criticlr002_trial
