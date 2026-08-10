#!/bin/bash
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --qos=cm4_serial
#SBATCH --job-name=sil_c1_pp_bi400_ss200
#SBATCH --output=sil_c1_pp_bi400_ss200_%j.log
#SBATCH --error=sil_c1_pp_bi400_ss200_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

cd "$HOME/Reduce-then-optimize" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
chmod +x run_sil_training_bi400_ss200_class1_pairpruner_v1final.sh
./run_sil_training_bi400_ss200_class1_pairpruner_v1final.sh
