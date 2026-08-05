#!/bin/bash
#SBATCH --job-name=wandb_gcn_l1
#SBATCH --output=wandb_gcn_l1_%j.log
#SBATCH --error=wandb_gcn_l1_%j.err
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --qos=cm4_serial
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
# 2026-08-05: generic wandb sweep agent job - each submission runs exactly
# one trial (--count 1), so scaling out just means submitting this same
# sbatch file multiple times rather than generating one file per HP combo.
./venv/bin/wandb agent --count 1 jonathan-homafar-technical-university-of-munich/gcn-l1-sweep/np8vurt5
