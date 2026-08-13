#!/bin/bash
#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --qos=cm4_serial_long
#SBATCH --job-name=sil_mc_rp_bi400_ss200_s42
#SBATCH --output=sil_mc_rp_bi400_ss200_s42_%j.log
#SBATCH --error=sil_mc_rp_bi400_ss200_s42_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

cd "$HOME/Reduce-then-optimize" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
chmod +x run_sil_training_bi400_ss200_mixedclass_requestpruner.sh
./run_sil_training_bi400_ss200_mixedclass_requestpruner.sh 42
