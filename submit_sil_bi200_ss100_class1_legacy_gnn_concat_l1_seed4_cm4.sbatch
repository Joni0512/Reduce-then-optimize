#!/bin/bash
#SBATCH --job-name=sil_c1_leg_gnnC1L_s4
#SBATCH --clusters=cm4
#SBATCH --partition=cm4_tiny
#SBATCH --qos=cm4_tiny
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=17
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=sil_c1_leg_gnnC1L_s4_%j.log
#SBATCH --error=sil_c1_leg_gnnC1L_s4_%j.err

# 2026-08-02: parallel-execution test - submitted standalone (no dependency),
# on cm4/cm4_tiny instead of serial, while the serial-cluster concat chain
# (submit_gnn_concat_seeds_chain.sh) is running. Goal: check whether this runs
# concurrently with the serial jobs, or fails/queues due to the Gurobi
# concurrent-session license limit regardless of cluster.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
chmod +x run_sil_training_bi200_ss100_class1_legacy_gnn_concat_l1_seed4.sh
./run_sil_training_bi200_ss100_class1_legacy_gnn_concat_l1_seed4.sh
