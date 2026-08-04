#!/bin/bash
#SBATCH --job-name=sil_hpp_1e3h128d0
#SBATCH --output=sil_hpp_1e3h128d0_%j.log
#SBATCH --error=sil_hpp_1e3h128d0_%j.err
#SBATCH --clusters=inter
#SBATCH --partition=cm4_inter
#SBATCH --qos=normal
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
chmod +x run_sil_training_bi200_ss100_class1_legacy_gnn_pool_l1_hp_lr1e3_h128_d0_seed1.sh
./run_sil_training_bi200_ss100_class1_legacy_gnn_pool_l1_hp_lr1e3_h128_d0_seed1.sh
