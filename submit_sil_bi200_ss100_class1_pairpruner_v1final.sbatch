#!/bin/bash
#SBATCH --job-name=sil_c1_pp_bi200_ss100
#SBATCH --output=sil_c1_pp_bi200_ss100_%j.log
#SBATCH --error=sil_c1_pp_bi200_ss100_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=cm4_inter

cd "$HOME/Reduce-then-optimize" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
chmod +x run_sil_training_bi200_ss100_class1_pairpruner_v1final.sh
./run_sil_training_bi200_ss100_class1_pairpruner_v1final.sh
