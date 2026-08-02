#!/bin/bash
#SBATCH --job-name=sil_c1_leg_gnnC2L_s4
#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --qos=cm4_serial_long
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=sil_c1_leg_gnnC2L_s4_%j.log
#SBATCH --error=sil_c1_leg_gnnC2L_s4_%j.err

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
chmod +x run_sil_training_bi200_ss100_class1_legacy_gnn_concat_l2_seed4.sh
./run_sil_training_bi200_ss100_class1_legacy_gnn_concat_l2_seed4.sh
