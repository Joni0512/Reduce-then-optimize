#!/bin/bash
#SBATCH --job-name=bc_sync_rho600
#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --qos=cm4_serial_long
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=bc_sync_rho600_%j.log
#SBATCH --error=bc_sync_rho600_%j.err

# Actor/RHO state-sync behavior cloning, RHO at bi600/ss100 (actor stays bi200/ss100) - see
# chat. Requires the logger.py handler-leak fix (2026-09-26) and the payload_parser.py
# state-sync fixes to already be in the checked-out code.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"

# Filtered through grep as a safety net against runaway log growth (see chat, disk-quota
# history) - only keep our own progress lines, not the raw per-iteration solver spam.
./venv/bin/python3 -u -m rtv_solver.pipeline.srl_behavior_cloning_state_sync_rho600 2>&1 | grep --line-buffered -E "^===|rho_service_rate|state_sync_loop_rho600|SKIPPING|DONE" > bc_sync_rho600_filtered.log
