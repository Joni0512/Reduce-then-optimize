#!/bin/bash
#SBATCH --job-name=srl_twin30
#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --qos=cm4_serial_long
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=srl_twin30_%j.log
#SBATCH --error=srl_twin30_%j.err

# 30-epoch twin-critic, exact same config as the best local run (twin_critic_epochs30_1,
# best_val=0.7085) except SEED - see chat, checking robustness across seeds.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"

if [ -z "$SEED" ]; then
    echo "SEED not set - submit with: sbatch --export=SEED=<n> submit_srl_cluster_twin_critic_epochs30_trial.sbatch" >&2
    exit 1
fi

./venv/bin/python3 -u -m rtv_solver.pipeline.srl_cluster_twin_critic_epochs30_trial "$SEED"
