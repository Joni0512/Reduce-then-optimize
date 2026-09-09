#!/bin/bash
#SBATCH --job-name=srl_tdboot_bi400ss200_12i
#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --qos=cm4_serial_long
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=srl_tdboot_bi400ss200_12i_%j.log
#SBATCH --error=srl_tdboot_bi400ss200_12i_%j.err

# 2026-09-10: TD-bootstrap (gamma=0.99, tau=0.005) at bi400/ss200 on the
# full balanced 12-instance test set (seed 42) - see chat. Follow-up to
# test_srl_td_bootstrap_bi400_ss200.py's 4-instance result, where lc108
# (the stable control at bi200/ss100) unexpectedly ALSO collapsed to 0.0 at
# bi400/ss200 - checking how general that new lc-instance collapse is.
#
# python3 -u (unbuffered) + grep --line-buffered - Python fully buffers
# stdout when piped, so print() output can sit unflushed for hours and be
# lost entirely if the job gets killed before flushing.
#
# Prerequisite: outputs/outputs/sil_training_bi200_ss100_mixed_balanced_legacy_mlp_seed1/
# (the SIL actor checkpoint) must already exist on the cluster.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
./venv/bin/python3 -u -m rtv_solver.pipeline.run_srl_balanced_td_bootstrap_bi400_ss200_12instances 2>&1 | grep --line-buffered -E "^===|^!!!|epoch [0-9]+ done" > filtered_run_srl_balanced_td_bootstrap_bi400_ss200_12instances.log
