#!/bin/bash
#SBATCH --job-name=srl_twin_t1
#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --qos=cm4_serial_long
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=srl_twin_tau0001_%j.log
#SBATCH --error=srl_twin_tau0001_%j.err

# 2026-09-25: twin-critic test at tau=0.001 (the original buggy value, see chat) - deliberate,
# for clean comparison against sandy-sweep-60 (single-critic, tau=0.001, best_val=0.6911) and
# against the tau=0.005 twin-critic/single-critic pair already run locally. SEED is exported by
# the submitter.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"

if [ -z "$SEED" ]; then
    echo "SEED not set - submit with: sbatch --export=SEED=<n> submit_srl_cluster_twin_critic_tau0001_trial.sbatch" >&2
    exit 1
fi

./venv/bin/python3 -u -m rtv_solver.pipeline.srl_cluster_twin_critic_tau0001_trial "$SEED"
