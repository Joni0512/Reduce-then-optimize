#!/bin/bash
#SBATCH --job-name=srl_tc_div12_seeds
#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --qos=cm4_serial_long
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=srl_tc_div12_seeds_%j.log
#SBATCH --error=srl_tc_div12_seeds_%j.err

# 2026-09-04: replicates the 12-instance target_critic Polyak (tau=0.3)
# divergence diagnostic across 4 additional seeds (43-46) - see chat and
# rtv_solver/pipeline/replicate_targetcritic_polyak03_diagnostic_12instances_seeds.py's
# own docstring. 4 seeds x 12 instances = 48 SRL runs plus 4 critic
# pretrains - on serial_long (48h) given the scale, and to avoid repeating
# the earlier time-limit-kill data loss (see chat: v1-top10-seeds and
# v2-top5-seeds both got killed on serial_std's 24h limit).
#
# python3 -u (unbuffered) + grep --line-buffered - see chat: Python fully
# buffers stdout when piped, so print() output can sit unflushed for hours
# and be lost entirely if the job gets killed before flushing.
#
# Prerequisite: outputs/outputs/sil_training_bi200_ss100_mixed_balanced_legacy_mlp_seed1/
# (the SIL actor checkpoint) must already exist on the cluster.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
./venv/bin/python3 -u -m rtv_solver.pipeline.replicate_targetcritic_polyak03_diagnostic_12instances_seeds 2>&1 | grep --line-buffered -E "^===|^!!!|epoch [0-9]+ done" > filtered_replicate_targetcritic_polyak03_diagnostic_12instances_seeds.log
