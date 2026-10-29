#!/bin/bash
#SBATCH --job-name=rh_base
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --qos=cm4_serial
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=16G
#SBATCH --output=rh_base_%j.log
#SBATCH --error=rh_base_%j.err

# 2026-10-29: RHO-no-learning baseline (label source of the request pruner) for one window and one shard. BI, SS, SHARD, NSHARDS exported by the submitter.
# serial_std (24 h, QOS cm4_serial: cpu=192, no memory limit). Resumable: re-submit the same job to continue (finished instances are skipped).
cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
./run_baseline_data_window.sh "$BI" "$SS" "${SHARD:-0}" "${NSHARDS:-1}"
