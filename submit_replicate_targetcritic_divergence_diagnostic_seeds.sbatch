#!/bin/bash
#SBATCH --job-name=srl_tc_div_seeds
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --qos=cm4_serial
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=srl_tc_div_seeds_%j.log
#SBATCH --error=srl_tc_div_seeds_%j.err

# 2026-09-03: replicates the target_critic vs live critic divergence
# diagnostic (Polyak tau=0.3, lrc207) across 4 additional seeds (43-46) -
# see chat and rtv_solver/pipeline/replicate_targetcritic_divergence_diagnostic_seeds.py's
# own docstring. Output filtered through grep - a previous unfiltered run's
# .err file grew to 17.5GB and exceeded the shared cluster disk quota,
# killing every other concurrently running job too.
#
# Prerequisite: outputs/outputs/sil_training_bi200_ss100_mixed_balanced_legacy_mlp_seed1/
# (the SIL actor checkpoint) must already exist on the cluster.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
# 2026-09-03: -u (unbuffered) + grep --line-buffered - see chat. Python
# fully buffers stdout when piped (not a tty), so print() output can sit
# unflushed for hours; if the job is then killed (e.g. SLURM time limit),
# everything buffered is lost - explains two earlier jobs finishing with
# completely empty filtered logs despite running ~24h. Also fixed the
# "epoch done" pattern (never matched the actual "epoch N done" text).
./venv/bin/python3 -u -m rtv_solver.pipeline.replicate_targetcritic_divergence_diagnostic_seeds 2>&1 | grep --line-buffered -E "^===|^!!!|epoch [0-9]+ done" > filtered_replicate_targetcritic_divergence_diagnostic_seeds.log
