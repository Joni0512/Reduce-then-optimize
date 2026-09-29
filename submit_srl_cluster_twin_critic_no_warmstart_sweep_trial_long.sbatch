#!/bin/bash
#SBATCH --job-name=srl_nows_swl
#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --qos=cm4_serial_long
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=srl_nows_swl_%j.log
#SBATCH --error=srl_nows_swl_%j.err

# serial_long variant of submit_srl_cluster_twin_critic_no_warmstart_sweep_trial.sbatch - see
# chat: 30-epoch no-warmstart runs took ~1h/epoch on serial_std, risking the 24h limit; moved
# to serial_long (7 days, but subject to the cm4_serial_long QOSMaxMemoryPerUser cap again).
# RUN_ID/ACTOR_LR/CRITIC_LR/SEED exported by the submitter.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"

./venv/bin/python3 -u -m rtv_solver.pipeline.srl_cluster_twin_critic_no_warmstart_sweep_trial "$RUN_ID" "$ACTOR_LR" "$CRITIC_LR" "$SEED"
