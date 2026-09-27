#!/bin/bash
#SBATCH --job-name=srl_nows_clr
#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --qos=cm4_serial_long
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=srl_nows_clr_%j.log
#SBATCH --error=srl_nows_clr_%j.err

# No-warmstart twin-critic, EPOCHS=30, seed=42, critic_lr swept - see chat. CRITIC_LR exported
# by the submitter.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"

if [ -z "$CRITIC_LR" ]; then
    echo "CRITIC_LR not set - submit with: sbatch --export=CRITIC_LR=<n> submit_srl_cluster_twin_critic_no_warmstart_epochs30_criticlr_trial.sbatch" >&2
    exit 1
fi

./venv/bin/python3 -u -m rtv_solver.pipeline.srl_cluster_twin_critic_no_warmstart_epochs30_criticlr_trial "$CRITIC_LR"
