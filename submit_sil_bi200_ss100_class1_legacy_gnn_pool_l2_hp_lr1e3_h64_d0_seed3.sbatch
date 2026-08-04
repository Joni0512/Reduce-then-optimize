#!/bin/bash
#SBATCH --job-name=sil_hppl2_s3
#SBATCH --output=sil_hppl2_s3_%j.log
#SBATCH --error=sil_hppl2_s3_%j.err
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --qos=cm4_serial
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
chmod +x run_sil_training_bi200_ss100_class1_legacy_gnn_pool_l2_hp_lr1e3_h64_d0_seed3.sh
./run_sil_training_bi200_ss100_class1_legacy_gnn_pool_l2_hp_lr1e3_h64_d0_seed3.sh
