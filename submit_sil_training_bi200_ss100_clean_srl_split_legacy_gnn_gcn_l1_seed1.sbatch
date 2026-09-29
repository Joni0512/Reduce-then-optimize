#!/bin/bash
#SBATCH --job-name=sil_clean_gnn
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --qos=cm4_serial
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=sil_clean_gnn_%j.log
#SBATCH --error=sil_clean_gnn_%j.err

# Clean SIL checkpoint (no VAL/TEST leakage), GNN GCN L1 variant - see chat.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"

chmod +x run_sil_training_bi200_ss100_clean_srl_split_legacy_gnn_gcn_l1_seed1.sh
./run_sil_training_bi200_ss100_clean_srl_split_legacy_gnn_gcn_l1_seed1.sh
