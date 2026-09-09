#!/bin/bash
#SBATCH --job-name=srl_tdboot_bi400ss200_seeds
#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --qos=cm4_serial_long
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=srl_tdboot_bi400ss200_seeds_%j.log
#SBATCH --error=srl_tdboot_bi400ss200_seeds_%j.err

# 2026-09-09: replicates the bi400/ss200 TD-bootstrap chain-length test
# (seed=42 run locally) across 4 additional seeds (43-46) - see chat.
# Same 4 instances (lrc207, lr210, lrc108, lc108), gamma=0.99, tau=0.005.
#
# python3 -u (unbuffered) + grep --line-buffered - Python fully buffers
# stdout when piped, so print() output can sit unflushed for hours and be
# lost entirely if the job gets killed before flushing.
#
# Prerequisite: outputs/outputs/sil_training_bi200_ss100_mixed_balanced_legacy_mlp_seed1/
# (the SIL actor checkpoint) must already exist on the cluster.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
./venv/bin/python3 -u -m rtv_solver.pipeline.replicate_td_bootstrap_bi400_ss200_seeds 2>&1 | grep --line-buffered -E "^===|^!!!|epoch [0-9]+ done" > filtered_replicate_td_bootstrap_bi400_ss200_seeds.log
