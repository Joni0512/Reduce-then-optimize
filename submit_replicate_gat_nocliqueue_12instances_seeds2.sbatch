#!/bin/bash
#SBATCH --job-name=srl_gat_nocq2
#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --qos=cm4_serial_long
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=srl_gat_nocq2_%j.log
#SBATCH --error=srl_gat_nocq2_%j.err

# 2026-09-06: second batch of seeds (47-51) for the GAT-nocliqueue test,
# running in PARALLEL with the original seeds-42-46 job (close to its 24h
# serial_std limit) - see chat. Uses serial_long (48h) since the earlier
# QOSMaxMemoryPerUser block should have cleared by now as the 3 older
# serial_long jobs finish. --mem=16G (see chat, same reasoning as the first
# GAT jobs - this workload doesn't need 32G).
#
# python3 -u (unbuffered) + grep --line-buffered - Python fully buffers
# stdout when piped, so print() output can sit unflushed for hours and be
# lost entirely if the job gets killed before flushing.
#
# Prerequisite: outputs/outputs/sil_training_bi200_ss100_mixed_balanced_legacy_mlp_seed1/
# (the SIL actor checkpoint) must already exist on the cluster.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
./venv/bin/python3 -u -m rtv_solver.pipeline.replicate_gat_nocliqueue_12instances_seeds2 2>&1 | grep --line-buffered -E "^===|^!!!|epoch [0-9]+ done" > filtered_replicate_gat_nocliqueue_12instances_seeds2.log
