#!/bin/bash
#SBATCH --job-name=srl_single_g
#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --qos=cm4_serial_long
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=srl_single_g_%j.log
#SBATCH --error=srl_single_g_%j.err

# Generic single-critic trial submitter - RUN_ID/REWARD_MODE/ACTOR_LR/CRITIC_LR/TAU/SEED
# exported by the submitter. Counterpart to submit_srl_cluster_twin_critic_generic_trial.sbatch,
# for same-seed single-vs-twin-critic comparisons.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"

if [ -z "$RUN_ID" ] || [ -z "$REWARD_MODE" ] || [ -z "$ACTOR_LR" ] || [ -z "$CRITIC_LR" ] || [ -z "$TAU" ] || [ -z "$SEED" ]; then
    echo "RUN_ID/REWARD_MODE/ACTOR_LR/CRITIC_LR/TAU/SEED not all set - see submit examples in chat" >&2
    exit 1
fi

./venv/bin/python3 -u -m rtv_solver.pipeline.srl_cluster_single_critic_generic_trial "$RUN_ID" "$REWARD_MODE" "$ACTOR_LR" "$CRITIC_LR" "$TAU" "$SEED"
