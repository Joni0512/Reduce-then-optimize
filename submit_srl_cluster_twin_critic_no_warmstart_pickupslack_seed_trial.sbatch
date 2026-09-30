#!/bin/bash
#SBATCH --job-name=srl_nows_pss
#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --qos=cm4_serial_long
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=srl_nows_pss_%j.log
#SBATCH --error=srl_nows_pss_%j.err

# Seed sweep of the best no-warmstart config so far (pickup_slack=True, baseline critic_lr,
# seed=42: best_val=0.7215/test=0.7702) - see chat. SEED exported by the submitter.
# serial_long chosen directly - no-warmstart 30-epoch runs take ~1h/epoch and a serial_std job
# (5634362) already hit the 24h TIMEOUT with no result.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"

./venv/bin/python3 -u -m rtv_solver.pipeline.srl_cluster_twin_critic_no_warmstart_pickupslack_seed_trial "$SEED"
