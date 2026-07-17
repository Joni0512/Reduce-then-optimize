#!/bin/bash
#SBATCH --job-name=sil_t06
#SBATCH --output=sil_t06_%j.log
#SBATCH --error=sil_t06_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --partition=cm4_inter

cd "$HOME/Reduce-then-optimize" || exit 1
chmod +x run_sil_training_bi400_ss200_t0.6.sh
./run_sil_training_bi400_ss200_t0.6.sh
