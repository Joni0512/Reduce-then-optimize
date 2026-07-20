#!/bin/bash
#SBATCH --job-name=sil_mixedclass_200_100
#SBATCH --output=sil_mixedclass_200_100_%j.log
#SBATCH --error=sil_mixedclass_200_100_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=cm4_inter

cd "$HOME/Reduce-then-optimize" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
chmod +x run_sil_training_bi200_ss100_mixedclass_t0.5.sh
./run_sil_training_bi200_ss100_mixedclass_t0.5.sh
