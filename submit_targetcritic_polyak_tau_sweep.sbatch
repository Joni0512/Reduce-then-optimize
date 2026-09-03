#!/bin/bash
#SBATCH --job-name=srl_tc_polyak_tau
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --qos=cm4_serial
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=srl_tc_polyak_tau_%j.log
#SBATCH --error=srl_tc_polyak_tau_%j.err

# 2026-08-31: target_critic Polyak tau grid (0.1-0.5), full balanced
# 12-instance set, seed=42 - see chat and
# rtv_solver/pipeline/sweep_targetcritic_polyak_tau_12instances.py's own
# docstring. Runs on the cluster alongside the other two seed-replication
# jobs (submit_replicate_v1_top10_seeds.sbatch,
# submit_replicate_targetcritic_replaybuffer_seeds.sbatch).
#
# Prerequisite: outputs/outputs/sil_training_bi200_ss100_mixed_balanced_legacy_mlp_seed1/
# (the SIL actor checkpoint) must already exist on the cluster.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
# 2026-09-02: filtered through grep instead of letting raw solver stdout/
# stderr flood the SBATCH --output/--error files - a previous run's .err
# file grew to 17.5GB and exceeded the cluster disk quota, killing every
# other concurrently running job too (shared quota) - see chat.
# 2026-09-03: -u (unbuffered) + grep --line-buffered - see chat.
./venv/bin/python3 -u -m rtv_solver.pipeline.sweep_targetcritic_polyak_tau_12instances 2>&1 | grep --line-buffered -E "^===|^!!!|tau=.*instance=|DONE" > filtered_targetcritic_polyak_tau_sweep.log
