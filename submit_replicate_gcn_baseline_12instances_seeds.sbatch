#!/bin/bash
#SBATCH --job-name=srl_gcn_base
#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --qos=cm4_serial_long
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=srl_gcn_base_%j.log
#SBATCH --error=srl_gcn_base_%j.err

# 2026-09-07: 5-seed gcn Replay-Buffer-alone baseline (seeds 42-46, matching
# the first GAT seed batch) - see chat. Replaces the single-seed 0.580
# baseline number with a properly matched multi-seed comparison against
# replicate_gat_nocliqueue_12instances_seeds.py's 0.658 (5-seed mean). gcn
# is faster than gat (~3.9h/seed observed vs ~7h/seed), so 48h should be
# more than enough for all 5 seeds.
#
# python3 -u (unbuffered) + grep --line-buffered - Python fully buffers
# stdout when piped, so print() output can sit unflushed for hours and be
# lost entirely if the job gets killed before flushing.
#
# Prerequisite: outputs/outputs/sil_training_bi200_ss100_mixed_balanced_legacy_mlp_seed1/
# (the SIL actor checkpoint) must already exist on the cluster.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
./venv/bin/python3 -u -m rtv_solver.pipeline.replicate_gcn_baseline_12instances_seeds 2>&1 | grep --line-buffered -E "^===|^!!!|epoch [0-9]+ done" > filtered_replicate_gcn_baseline_12instances_seeds.log
