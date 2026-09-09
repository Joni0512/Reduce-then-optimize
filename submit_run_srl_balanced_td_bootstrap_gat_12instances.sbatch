#!/bin/bash
#SBATCH --job-name=srl_tdboot_gat_12i
#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --qos=cm4_serial_long
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=srl_tdboot_gat_12i_%j.log
#SBATCH --error=srl_tdboot_gat_12i_%j.err

# 2026-09-10: TD-bootstrap (gamma=0.99, tau=0.005) combined with a GAT
# critic, on the full balanced 12-instance test set (seed 42) - see chat.
# The 4-instance test found GAT did NOT collapse on lrc207 under
# td_bootstrap (final SR 0.471, gcn collapsed to 0.000) - checking whether
# that holds set-wide. Part of the gcn/gat x monte_carlo/td_bootstrap 2x2
# comparison, all on the same 12 instances.
#
# python3 -u (unbuffered) + grep --line-buffered - Python fully buffers
# stdout when piped, so print() output can sit unflushed for hours and be
# lost entirely if the job gets killed before flushing.
#
# Prerequisite: outputs/outputs/sil_training_bi200_ss100_mixed_balanced_legacy_mlp_seed1/
# (the SIL actor checkpoint) must already exist on the cluster.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
./venv/bin/python3 -u -m rtv_solver.pipeline.run_srl_balanced_td_bootstrap_gat_12instances 2>&1 | grep --line-buffered -E "^===|^!!!|epoch [0-9]+ done" > filtered_run_srl_balanced_td_bootstrap_gat_12instances.log
