#!/bin/bash
#SBATCH --job-name=srl_tdboot_gat_locpos_12i
#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --qos=cm4_serial_long
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=srl_tdboot_gat_locpos_12i_%j.log
#SBATCH --error=srl_tdboot_gat_locpos_12i_%j.err

# 2026-09-10: TD-bootstrap (gamma=0.99, tau=0.005) with GAT critic AND
# reward_mode="local_positive" combined, on the full balanced 12-instance
# test set, seed=43 - see chat. Checks whether combining GAT + denser
# reward closes the remaining gcn+local_positive gap on lrc207/lr210, or
# whether GAT alone already captures the full effect. Local run covers the
# same config on 4 diagnostic instances at seed=42.
#
# python3 -u (unbuffered) + grep --line-buffered - Python fully buffers
# stdout when piped, so print() output can sit unflushed for hours and be
# lost entirely if the job gets killed before flushing.
#
# Prerequisite: outputs/outputs/sil_training_bi200_ss100_mixed_balanced_legacy_mlp_seed1/
# (the SIL actor checkpoint) must already exist on the cluster.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
./venv/bin/python3 -u -m rtv_solver.pipeline.run_srl_balanced_td_bootstrap_gat_local_positive_12instances_seed43 2>&1 | grep --line-buffered -E "^===|^!!!|epoch [0-9]+ done" > filtered_run_srl_balanced_td_bootstrap_gat_local_positive_12instances_seed43.log
