#!/bin/bash
#SBATCH --job-name=srl_gat_nocq
#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --qos=cm4_serial_long
#SBATCH --time=48:00:00
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
# 2026-09-06: --mem lowered from 32G to 16G (see chat) - 3 other jobs
# already running on serial_long (96GB total) hit QOSMaxMemoryPerUser and
# kept these 2 new jobs stuck PENDING; this workload (small Li&Lim-scale
# graphs, same order of magnitude as the other critic-training jobs) does
# not actually need 32G.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
./venv/bin/python3 -u -m rtv_solver.pipeline.replicate_gat_nocliqueue_12instances_seeds 2>&1 | grep --line-buffered -E "^===|^!!!|epoch [0-9]+ done" > filtered_replicate_gat_nocliqueue_12instances_seeds.log
