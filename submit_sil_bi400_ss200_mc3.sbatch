#!/bin/bash
#SBATCH --job-name=sil_mc3
#SBATCH --output=sil_mc3_%j.log
#SBATCH --error=sil_mc3_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=cm4_inter

cd "$HOME/Reduce-then-optimize" || exit 1
chmod +x run_sil_training_bi400_ss200_mc3_t0.5.sh
./run_sil_training_bi400_ss200_mc3_t0.5.sh
