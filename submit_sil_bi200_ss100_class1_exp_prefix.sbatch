#!/bin/bash
#SBATCH --job-name=sil_class1_expprefix
#SBATCH --output=sil_class1_expprefix_%j.log
#SBATCH --error=sil_class1_expprefix_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=cm4_inter

cd "$HOME/Reduce-then-optimize" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
chmod +x run_sil_training_bi200_ss100_class1_exp_prefix.sh
./run_sil_training_bi200_ss100_class1_exp_prefix.sh
