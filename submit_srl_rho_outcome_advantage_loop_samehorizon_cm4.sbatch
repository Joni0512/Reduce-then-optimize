#!/bin/bash
#SBATCH --job-name=srl_rho_adv_samehz
#SBATCH --clusters=cm4
#SBATCH --partition=cm4_tiny
#SBATCH --qos=cm4_tiny
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=srl_rho_adv_samehz_%j.log
#SBATCH --error=srl_rho_adv_samehz_%j.err

# 2026-09-14: same-horizon RHO-outcome-advantage variant (actor AND RHO both
# at bi200/ss100, see chat) on the cm4/cm4_tiny queue - separate from the
# bi200-vs-bi400 version already running on serial_long. cm4_tiny's
# confirmed max walltime (from existing repo scripts) is 8h, well short of
# what a full 30-epoch/38-instance run likely needs - this will probably
# get killed by the time limit before finishing all 30 epochs. Best-val
# checkpoints saved so far survive; the final CSV/plot (written only at the
# very end) will NOT be produced if the time limit is hit first.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"

./venv/bin/python3 -u -m rtv_solver.pipeline.srl_rho_outcome_advantage_loop_samehorizon
