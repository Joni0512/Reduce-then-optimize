#!/bin/bash
#SBATCH --job-name=wandb_pool_l2
#SBATCH --output=wandb_pool_l2_%j.log
#SBATCH --error=wandb_pool_l2_%j.err
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --qos=cm4_serial
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
# 2026-08-06: same pattern as the GCN L1 wandb sweep agent job - one trial
# per submission (--count 1), scale out by resubmitting this sbatch file.
# SWEEP_ID must be filled in after `wandb sweep sweep_pool_l2.yaml` on the cluster.
./venv/bin/wandb agent --count 1 jonathan-homafar-technical-university-of-munich/pool-l2-sweep/SWEEP_ID
