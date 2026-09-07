#!/bin/bash
#SBATCH --job-name=srl_card3_gcn
#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --qos=cm4_serial_long
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=srl_card3_gcn_%j.log
#SBATCH --error=srl_card3_gcn_%j.err

# 2026-09-07: tests max_cardinality=3 (vs. the Config default of 2 used by
# every SRL run this session so far), plain gcn aggregator, 5 seeds, all 12
# instances - see chat, motivated by Table 4 (RHO/SIL/OPT paper comparison)
# showing service rate rising from cardinality 2 to 3 to 4 at
# horizon=200/step=100. Companion job
# submit_replicate_gat_nocliqueue_cardinality3_seeds.sbatch tests the same
# change with aggregator="gat".
#
# python3 -u (unbuffered) + grep --line-buffered - Python fully buffers
# stdout when piped, so print() output can sit unflushed for hours and be
# lost entirely if the job gets killed before flushing.
#
# Prerequisite: outputs/outputs/sil_training_bi200_ss100_mixed_balanced_legacy_mlp_seed1/
# (the SIL actor checkpoint) must already exist on the cluster.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
./venv/bin/python3 -u -m rtv_solver.pipeline.replicate_cardinality3_12instances_seeds 2>&1 | grep --line-buffered -E "^===|^!!!|epoch [0-9]+ done" > filtered_replicate_cardinality3_12instances_seeds.log
