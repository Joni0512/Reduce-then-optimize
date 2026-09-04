#!/bin/bash
#SBATCH --job-name=srl_tc_polyak_fine
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --qos=cm4_serial
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=srl_tc_polyak_fine_%j.log
#SBATCH --error=srl_tc_polyak_fine_%j.err

# 2026-09-04: fine-grained target_critic Polyak tau grid [0.002, 0.003,
# 0.004], full balanced 12-instance set, seed=42 - see chat and
# rtv_solver/pipeline/sweep_targetcritic_polyak_tau_fine_12instances.py's
# own docstring. Narrows the gap between tau=0.001 (worse) and tau=0.005
# (best result so far).
#
# python3 -u (unbuffered) + grep --line-buffered - Python fully buffers
# stdout when piped, so print() output can sit unflushed for hours and be
# lost entirely if the job gets killed before flushing.
#
# Prerequisite: outputs/outputs/sil_training_bi200_ss100_mixed_balanced_legacy_mlp_seed1/
# (the SIL actor checkpoint) must already exist on the cluster.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
./venv/bin/python3 -u -m rtv_solver.pipeline.sweep_targetcritic_polyak_tau_fine_12instances 2>&1 | grep --line-buffered -E "^===|^!!!|tau=.*instance=|DONE" > filtered_targetcritic_polyak_tau_fine_sweep.log
