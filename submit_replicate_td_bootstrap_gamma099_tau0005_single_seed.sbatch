#!/bin/bash
#SBATCH --job-name=srl_tdboot_seed
#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --qos=cm4_serial_long
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=srl_tdboot_seed_%j.log
#SBATCH --error=srl_tdboot_seed_%j.err

# 2026-09-10: single-seed TD-bootstrap (gamma=0.99, tau=0.005) 12-instance
# run - lets seeds run as separate PARALLEL cluster jobs instead of one
# sequential job looping over all of them (5482637 covers 43-46
# sequentially) - see chat.
#
# Usage: sbatch --export=SEED=45 submit_replicate_td_bootstrap_gamma099_tau0005_single_seed.sbatch
#
# python3 -u (unbuffered) + grep --line-buffered - Python fully buffers
# stdout when piped, so print() output can sit unflushed for hours and be
# lost entirely if the job gets killed before flushing.
#
# Prerequisite: outputs/outputs/sil_training_bi200_ss100_mixed_balanced_legacy_mlp_seed1/
# (the SIL actor checkpoint) must already exist on the cluster.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"

if [ -z "$SEED" ]; then
    echo "SEED not set - pass it via: sbatch --export=SEED=45 $0"
    exit 1
fi

./venv/bin/python3 -u -m rtv_solver.pipeline.replicate_td_bootstrap_gamma099_tau0005_single_seed "$SEED" 2>&1 | grep --line-buffered -E "^===|^!!!|epoch [0-9]+ done" > "filtered_replicate_td_bootstrap_gamma099_tau0005_seed${SEED}.log"
