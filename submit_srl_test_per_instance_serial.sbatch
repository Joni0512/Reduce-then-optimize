#!/bin/bash
#SBATCH --job-name=srl_test_pi
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --qos=cm4_serial
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=srl_test_pi_%j.log
#SBATCH --error=srl_test_pi_%j.err

# 2026-10-09: per-instance TEST rates for all buffered SRL runs on the cluster (actor_lr scan runs, serial/cm4 sweep trials), recomputed from
# each run's best_actor_checkpoint.pt - see scripts/analysis/compute_srl_test_per_instance.py (skips runs that already have a JSON).
# ~1.5 min per run on the Mac, expect ~5x on the cluster (about 8 min per run). Results: outputs/srl_test_per_instance/*.json and one
# "PERINST ..." line per run in this job's log. Not run on the login node (LRZ rule: no extensive computation there).

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
./venv/bin/python3 -u scripts/analysis/compute_srl_test_per_instance.py "$SLURM_SUBMIT_DIR/outputs/srl_test_per_instance/_eval_dirs"
