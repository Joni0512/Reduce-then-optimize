#!/bin/bash
#SBATCH --job-name=sil_c1_exp_bi100s50
#SBATCH --output=sil_c1_exp_bi100s50_%j.log
#SBATCH --error=sil_c1_exp_bi100s50_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=cm4_inter

cd "$HOME/Reduce-then-optimize" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
chmod +x run_sil_training_bi100_ss50_class1_exp_prefix.sh
./run_sil_training_bi100_ss50_class1_exp_prefix.sh
