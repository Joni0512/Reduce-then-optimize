#!/bin/bash
#SBATCH --job-name=srl_tc_rb_replicate_seeds
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --qos=cm4_serial
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=srl_tc_rb_replicate_seeds_%j.log
#SBATCH --error=srl_tc_rb_replicate_seeds_%j.err

# 2026-08-31: replicates the target_critic (X=5) + replay-buffer combined
# variant across 4 additional seeds (43-46) - see chat and
# rtv_solver/pipeline/replicate_targetcritic_replaybuffer_seeds.py's own
# docstring. Companion job to submit_replicate_v1_top10_seeds.sbatch - both
# run concurrently on the cluster, independent of each other and of the
# local machine's v2 wandb sweep.
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
# pattern (never matched actual "epoch N done" text) - see chat.
./venv/bin/python3 -u -m rtv_solver.pipeline.replicate_targetcritic_replaybuffer_seeds 2>&1 | grep --line-buffered -E "^===|^!!!|epoch [0-9]+ done|seed=.*instance=|DONE" > filtered_replicate_targetcritic_replaybuffer_seeds.log
