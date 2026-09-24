#!/bin/bash
#SBATCH --job-name=srl_twin_critic
#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --qos=cm4_serial_long
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=srl_twin_critic_%j.log
#SBATCH --error=srl_twin_critic_%j.err

# 2026-09-24: twin-critic (TD3-style, min-target/mean-action-selection, see
# chat) first real test on the cluster, full 38-train/9-val split, fixed
# hyperparameters (best trial found so far in the buggy tau=0.001 sweep:
# sandy-sweep-60, actor_lr=0.00319, critic_lr=0.00108, reward_mode=
# local_positive), corrected to tau=0.005/gamma=0.99. SEED is exported by
# the submitter, different from the local run's seed=42, so this is an
# independent-seed replication, not a duplicate of the local test.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"

if [ -z "$SEED" ]; then
    echo "SEED not set - submit with: sbatch --export=SEED=<n> submit_srl_cluster_twin_critic_trial.sbatch" >&2
    exit 1
fi

./venv/bin/python3 -u -m rtv_solver.pipeline.srl_cluster_twin_critic_trial "$SEED"
