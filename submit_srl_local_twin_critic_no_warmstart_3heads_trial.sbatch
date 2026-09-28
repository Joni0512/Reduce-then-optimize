#!/bin/bash
#SBATCH --job-name=srl_3heads
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --qos=cm4_serial
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=srl_3heads_%j.log
#SBATCH --error=srl_3heads_%j.err

# No-warmstart twin-critic, gat_num_heads=3, 30 epochs, seed=42 - see chat, local run of this
# died unexpectedly (no DONE/FAILED, process just vanished) so retrying on the cluster instead.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"

./venv/bin/python3 -u -m rtv_solver.pipeline.srl_local_twin_critic_no_warmstart_3heads_trial
