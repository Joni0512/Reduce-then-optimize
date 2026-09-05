#!/bin/bash
#SBATCH --job-name=srl_gat_nocq
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --qos=cm4_serial
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=srl_gat_nocq_%j.log
#SBATCH --error=srl_gat_nocq_%j.err

# 2026-09-06: first multi-seed GAT test (aggregator="gat"), WITHOUT the
# route-clique edge extension, baseline hyperparameters, 5 seeds, all 12
# instances - see chat and docs/SRL_Design.md's GAT plan section. Companion
# job submit_replicate_gat_routeclique_12instances_seeds.sbatch tests WITH
# the clique extension for comparison.
#
# python3 -u (unbuffered) + grep --line-buffered - Python fully buffers
# stdout when piped, so print() output can sit unflushed for hours and be
# lost entirely if the job gets killed before flushing.
#
# Prerequisite: outputs/outputs/sil_training_bi200_ss100_mixed_balanced_legacy_mlp_seed1/
# (the SIL actor checkpoint) must already exist on the cluster.
#
# 2026-09-06: --mem lowered from 32G to 16G, AND switched from
# serial_long/cm4_serial_long to serial_std/cm4_serial (see chat) - 3 other
# jobs already running on serial_long (96GB total) hit QOSMaxMemoryPerUser
# and kept these 2 new jobs stuck PENDING; serial_std runs under a separate
# QOS with its own memory quota (unaffected by the serial_long jobs) and had
# 8 idle nodes at the time. Trade-off: 24h time limit instead of 48h - 5
# seeds x 12 instances took ~15-20h in comparable prior runs, should usually
# fit but is closer to the wall-clock cap than serial_long was.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
./venv/bin/python3 -u -m rtv_solver.pipeline.replicate_gat_nocliqueue_12instances_seeds 2>&1 | grep --line-buffered -E "^===|^!!!|epoch [0-9]+ done" > filtered_replicate_gat_nocliqueue_12instances_seeds.log
