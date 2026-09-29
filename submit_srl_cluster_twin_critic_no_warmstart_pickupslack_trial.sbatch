#!/bin/bash
#SBATCH --job-name=srl_nows_ps
#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --qos=cm4_serial_long
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=srl_nows_ps_%j.log
#SBATCH --error=srl_nows_ps_%j.err

# No-warmstart twin-critic, seed=42, 30 epochs, with ENABLE_PICKUP_SLACK_FEATURE=True - see
# chat and srl_cluster_twin_critic_no_warmstart_pickupslack_trial.py's own docstring. Same
# actor_lr/critic_lr as the best known no-warmstart config (val=0.6872 at 30 epochs), only the
# feature set differs. serial_long chosen directly (not serial_std) because the equivalent
# no-pickup-slack no-warmstart 30-epoch runs took ~1h/epoch on serial_std, risking its 24h cap.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"

./venv/bin/python3 -u -m rtv_solver.pipeline.srl_cluster_twin_critic_no_warmstart_pickupslack_trial
