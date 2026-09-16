#!/bin/bash
#SBATCH --job-name=bc_vs_rho_38i
#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --qos=cm4_serial_long
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=srl_bc_vs_rho_%j.log
#SBATCH --error=srl_bc_vs_rho_%j.err

# 2026-09-16: Behavior Cloning (Option 1) against a live RHO baseline, RHO
# at bi400/ss100 (longer horizon than the actor's bi200/ss100 - original
# design intent, see srl_behavior_cloning_vs_rho_loop.py's module docstring:
# matching horizons was tried and neither fixed the ManifestConsistencyError
# nor was it the intended comparison), on the full 38-instance train split -
# sanity check requested by the advisor: can the actor imitate a
# more-informed RHO at all (see chat, 2026-09-16). ManifestConsistencyError
# occurrences are caught and that instance skipped (~18% of train instances
# in the same-horizon run, expect similar here). RHO baselines are
# precomputed once per instance and cached (38 solves total), not
# recomputed per epoch.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"

./venv/bin/python3 -u -m rtv_solver.pipeline.srl_behavior_cloning_vs_rho_loop
