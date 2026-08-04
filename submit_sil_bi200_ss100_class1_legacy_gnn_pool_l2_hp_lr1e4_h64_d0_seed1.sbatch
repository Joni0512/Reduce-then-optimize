#!/bin/bash
#SBATCH --job-name=sil_hppl2_1e4
#SBATCH --output=sil_hppl2_1e4_%j.log
#SBATCH --error=sil_hppl2_1e4_%j.err
#SBATCH --clusters=cm4
#SBATCH --partition=cm4_tiny
#SBATCH --qos=cm4_tiny
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=17
#SBATCH --mem=32G

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
chmod +x run_sil_training_bi200_ss100_class1_legacy_gnn_pool_l2_hp_lr1e4_h64_d0_seed1.sh
./run_sil_training_bi200_ss100_class1_legacy_gnn_pool_l2_hp_lr1e4_h64_d0_seed1.sh
