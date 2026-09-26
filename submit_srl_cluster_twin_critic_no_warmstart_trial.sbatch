#!/bin/bash
#SBATCH --job-name=srl_twin_nows
#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --qos=cm4_serial_long
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=srl_twin_nows_%j.log
#SBATCH --error=srl_twin_nows_%j.err

# No-warmstart twin-critic (actor randomly initialized, no SIL checkpoint) - see chat, matches
# reference SRL paper's "same random model initialization for SIL, PPO, and SRL". SEED exported
# by the submitter.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"

if [ -z "$SEED" ]; then
    echo "SEED not set - submit with: sbatch --export=SEED=<n> submit_srl_cluster_twin_critic_no_warmstart_trial.sbatch" >&2
    exit 1
fi

./venv/bin/python3 -u -m rtv_solver.pipeline.srl_cluster_twin_critic_no_warmstart_trial "$SEED"
