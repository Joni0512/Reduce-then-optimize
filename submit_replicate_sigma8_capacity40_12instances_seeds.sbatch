#!/bin/bash
#SBATCH --job-name=srl_sig8_cap40
#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --qos=cm4_serial_long
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=srl_sig8_cap40_%j.log
#SBATCH --error=srl_sig8_cap40_%j.err

# 2026-09-05: tests sigma=8 (high perturbation) at replay_capacity=40 (the
# original best buffer size) across 5 seeds, all 12 instances - see chat.
# Checks whether the local v4 wandb sweep's high-sigma finding (mean final
# SR 0.577 at sigma~8.5, capacity=1000) transfers to capacity=40 too, or is
# specific to the larger buffer.
#
# python3 -u (unbuffered) + grep --line-buffered - Python fully buffers
# stdout when piped, so print() output can sit unflushed for hours and be
# lost entirely if the job gets killed before flushing.
#
# Prerequisite: outputs/outputs/sil_training_bi200_ss100_mixed_balanced_legacy_mlp_seed1/
# (the SIL actor checkpoint) must already exist on the cluster.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
./venv/bin/python3 -u -m rtv_solver.pipeline.replicate_sigma8_capacity40_12instances_seeds 2>&1 | grep --line-buffered -E "^===|^!!!|epoch [0-9]+ done" > filtered_replicate_sigma8_capacity40_12instances_seeds.log
