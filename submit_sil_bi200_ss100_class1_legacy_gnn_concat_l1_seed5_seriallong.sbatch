#!/bin/bash
#SBATCH --job-name=sil_c1_leg_gnnC1L_s5
#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --qos=cm4_serial
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=sil_c1_leg_gnnC1L_s5_%j.log
#SBATCH --error=sil_c1_leg_gnnC1L_s5_%j.err

# 2026-08-02: third-queue parallel-execution test - serial_long instead of
# serial_std, submitted standalone while serial_std (chain job) and cm4_tiny
# (seed4) are already running concurrently.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
chmod +x run_sil_training_bi200_ss100_class1_legacy_gnn_concat_l1_seed5.sh
./run_sil_training_bi200_ss100_class1_legacy_gnn_concat_l1_seed5.sh
