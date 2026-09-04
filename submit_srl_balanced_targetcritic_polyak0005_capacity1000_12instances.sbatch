#!/bin/bash
#SBATCH --job-name=srl_tc005_cap1000
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --qos=cm4_serial
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=srl_tc005_cap1000_%j.log
#SBATCH --error=srl_tc005_cap1000_%j.err

# 2026-09-04: replay_capacity=1000 (de facto unbounded) COMBINED with
# target_critic Polyak tau=0.005 (best tau found so far), full balanced
# 12-instance test set - see chat. New, previously-untested combination.
#
# python3 -u (unbuffered) + grep --line-buffered - Python fully buffers
# stdout when piped, so print() output can sit unflushed for hours and be
# lost entirely if the job gets killed before flushing.
#
# Prerequisite: outputs/outputs/sil_training_bi200_ss100_mixed_balanced_legacy_mlp_seed1/
# (the SIL actor checkpoint) must already exist on the cluster.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
./venv/bin/python3 -u -m rtv_solver.pipeline.run_srl_balanced_targetcritic_polyak0005_capacity1000_12instances 2>&1 | grep --line-buffered -E "^===|^!!!|epoch [0-9]+ done" > filtered_srl_tc005_cap1000.log
