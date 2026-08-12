#!/bin/bash
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --qos=cm4_serial
#SBATCH --job-name=sil_c1_rp_bi400_ss200_s2
#SBATCH --output=sil_c1_rp_bi400_ss200_s2_%j.log
#SBATCH --error=sil_c1_rp_bi400_ss200_s2_%j.err
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

cd "$HOME/Reduce-then-optimize" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
chmod +x run_sil_training_bi400_ss200_class1_requestpruner.sh
./run_sil_training_bi400_ss200_class1_requestpruner.sh 2
