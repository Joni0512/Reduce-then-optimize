#!/bin/bash
#SBATCH --job-name=srl_gat_nobootstrap
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --qos=cm4_serial
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=srl_gat_nobootstrap_%j.log
#SBATCH --error=srl_gat_nobootstrap_%j.err

# 2026-09-10: GAT critic WITHOUT TD-bootstrap (monte_carlo default target) -
# counterpart to test_srl_td_bootstrap_gat.py, same seed/instances/replay
# buffer config, so gcn/gat x monte_carlo/td_bootstrap can be compared
# cleanly - see chat.
#
# python3 -u (unbuffered) + grep --line-buffered - Python fully buffers
# stdout when piped, so print() output can sit unflushed for hours and be
# lost entirely if the job gets killed before flushing.
#
# Prerequisite: outputs/outputs/sil_training_bi200_ss100_mixed_balanced_legacy_mlp_seed1/
# (the SIL actor checkpoint) must already exist on the cluster.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
./venv/bin/python3 -u -m rtv_solver.pipeline.test_srl_gat_no_bootstrap 2>&1 | grep --line-buffered -E "^===|^!!!|epoch [0-9]+ done" > filtered_test_srl_gat_no_bootstrap.log
