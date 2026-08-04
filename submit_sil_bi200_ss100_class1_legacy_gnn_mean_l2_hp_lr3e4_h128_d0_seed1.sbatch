#!/bin/bash
#SBATCH --job-name=sil_hpm2_3e4h128d0
#SBATCH --output=sil_hpm2_3e4h128d0_%j.log
#SBATCH --error=sil_hpm2_3e4h128d0_%j.err
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --qos=cm4_serial
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
chmod +x run_sil_training_bi200_ss100_class1_legacy_gnn_mean_l2_hp_lr3e4_h128_d0_seed1.sh
./run_sil_training_bi200_ss100_class1_legacy_gnn_mean_l2_hp_lr3e4_h128_d0_seed1.sh
