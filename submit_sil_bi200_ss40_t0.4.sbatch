#!/bin/bash
#SBATCH --job-name=sil_t04_bi200_40
#SBATCH --output=sil_t04_bi200_40_%j.log
#SBATCH --error=sil_t04_bi200_40_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=cm4_inter

cd "$HOME/Reduce-then-optimize" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
chmod +x run_sil_training_bi200_ss40_t0.4.sh
./run_sil_training_bi200_ss40_t0.4.sh
