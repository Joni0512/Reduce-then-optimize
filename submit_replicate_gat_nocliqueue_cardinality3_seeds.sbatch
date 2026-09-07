#!/bin/bash
#SBATCH --job-name=srl_card3_gat
#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --qos=cm4_serial_long
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=srl_card3_gat_%j.log
#SBATCH --error=srl_card3_gat_%j.err

# 2026-09-07: tests max_cardinality=3 combined with aggregator="gat" (no
# route-clique), 5 seeds, all 12 instances - see chat. Companion job
# submit_replicate_cardinality3_12instances_seeds.sbatch tests the same
# cardinality change with plain gcn for comparison. GAT is ~1.9x slower
# than gcn per seed (observed this session, ~7h vs ~3.9h/seed) - budget
# accordingly, 48h may not be enough for all 5 seeds.
#
# python3 -u (unbuffered) + grep --line-buffered - Python fully buffers
# stdout when piped, so print() output can sit unflushed for hours and be
# lost entirely if the job gets killed before flushing.
#
# Prerequisite: outputs/outputs/sil_training_bi200_ss100_mixed_balanced_legacy_mlp_seed1/
# (the SIL actor checkpoint) must already exist on the cluster.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
./venv/bin/python3 -u -m rtv_solver.pipeline.replicate_gat_nocliqueue_cardinality3_seeds 2>&1 | grep --line-buffered -E "^===|^!!!|epoch [0-9]+ done" > filtered_replicate_gat_nocliqueue_cardinality3_seeds.log
