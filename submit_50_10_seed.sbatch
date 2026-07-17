#!/bin/bash
#SBATCH --job-name=rp_50_10_seed
#SBATCH --output=experiment_50_10_seed%a_%j.log
#SBATCH --error=experiment_50_10_seed%a_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --partition=cm4_inter

cd "$HOME/Reduce-then-optimize" || exit 1
chmod +x run_window_50_10_seed.sh
./run_window_50_10_seed.sh "$1"
