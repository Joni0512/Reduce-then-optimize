#!/bin/bash
#SBATCH --job-name=srl_tdboot_gat
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --qos=cm4_serial
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=srl_tdboot_gat_%j.log
#SBATCH --error=srl_tdboot_gat_%j.err

# 2026-09-09: TD-bootstrap critic target combined with a GAT critic
# (aggregator="gat") - not tested together before, all TD-bootstrap runs
# so far used the gcn critic - see chat. gamma=0.99, tau=0.005, on the 4
# diagnostic instances (lrc207, lr210, lrc108 - collapsed under
# td_bootstrap+gcn; lc108 - stable control).
#
# python3 -u (unbuffered) + grep --line-buffered - Python fully buffers
# stdout when piped, so print() output can sit unflushed for hours and be
# lost entirely if the job gets killed before flushing.
#
# Prerequisite: outputs/outputs/sil_training_bi200_ss100_mixed_balanced_legacy_mlp_seed1/
# (the SIL actor checkpoint) must already exist on the cluster.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
./venv/bin/python3 -u -m rtv_solver.pipeline.test_srl_td_bootstrap_gat 2>&1 | grep --line-buffered -E "^===|^!!!|epoch [0-9]+ done" > filtered_test_srl_td_bootstrap_gat.log
