#!/bin/bash
#SBATCH --job-name=rtv_final_compare
#SBATCH --partition=teramem_inter   # 10-day walltime cap, same partition as submit_overnight_mc3.sbatch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # matches MAX_THREAD_CNT=16 used by the multiprocessing pool
#SBATCH --mem=32G
#SBATCH --time=96:00:00             # generous: bi50_ss10 showed some instances can take 30min+ each,
                                     # with combinatorial-explosion risk on top (see lc207/lr211 pathologies)
#SBATCH --output=slurm_final_compare_%j.out
#SBATCH --error=slurm_final_compare_%j.err

# 2026-07-12: prepared while running the bi50_ss10 (short rolling-horizon window)
# comparison locally, which turned out ~1000x slower than bi400_ss100 on some
# instances -- moving future window sweeps to the cluster instead of tying up
# the local machine for potentially days.
#
# Usage: sbatch submit_final_compare_window.sbatch <batch_interval> <step_size> <threshold>
# Example: sbatch submit_final_compare_window.sbatch 50 10 0.5

set -e

cd "$SLURM_SUBMIT_DIR"

# TODO: activate the cluster environment (per README.md "Runtime on CPU cluster"):
# conda activate coaml
# (Gurobi/Hexaly license env vars, e.g. HX_LICENSE_PATH, must already be set in ~/.bashrc)

BATCH_INTERVAL="${1:?usage: sbatch submit_final_compare_window.sbatch <batch_interval> <step_size> <threshold>}"
STEP_SIZE="${2:?usage: sbatch submit_final_compare_window.sbatch <batch_interval> <step_size> <threshold>}"
THRESHOLD="${3:?usage: sbatch submit_final_compare_window.sbatch <batch_interval> <step_size> <threshold>}"

./run_final_compare_window.sh "$BATCH_INTERVAL" "$STEP_SIZE" "$THRESHOLD"
