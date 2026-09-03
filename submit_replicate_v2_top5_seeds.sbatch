#!/bin/bash
#SBATCH --job-name=srl_v2top5_replicate_seeds
#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --qos=cm4_serial_long
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=srl_v2top5_replicate_seeds_%j.log
#SBATCH --error=srl_v2top5_replicate_seeds_%j.err

# 2026-09-02: replicates the v2 wandb sweep's top-5 hyperparameter configs
# across 4 additional seeds (43-46), full balanced 12-instance test set -
# see chat and rtv_solver/pipeline/replicate_v2_top5_seeds.py's own
# docstring. User's call: cluster only, not local.
#
# Prerequisite: outputs/outputs/sil_training_bi200_ss100_mixed_balanced_legacy_mlp_seed1/
# (the SIL actor checkpoint) must already exist on the cluster.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
# 2026-09-02: filtered through grep instead of letting raw solver stdout/
# stderr flood the SBATCH --output/--error files - a previous run's .err
# file grew to 17.5GB and exceeded the cluster disk quota, killing every
# other concurrently running job too (shared quota) - see chat.
# 2026-09-03: -u (unbuffered) + grep --line-buffered, fixed "epoch done"
# pattern (never matched actual "epoch N done" text) - see chat. Also this
# job hit the 24h time limit on serial_std without finishing (240 runs: 4
# seeds x 5 configs x 12 instances) - switched to serial_long below.
./venv/bin/python3 -u -m rtv_solver.pipeline.replicate_v2_top5_seeds 2>&1 | grep --line-buffered -E "^===|^!!!|epoch [0-9]+ done|seed=.*config=|DONE" > filtered_replicate_v2_top5_seeds.log
