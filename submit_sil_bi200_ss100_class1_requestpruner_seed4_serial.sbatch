#!/bin/bash
#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --qos=cm4_serial_long
#SBATCH --job-name=sil_c1_rp_bi200_ss100_s4
#SBATCH --output=sil_c1_rp_bi200_ss100_s4_%j.log
#SBATCH --error=sil_c1_rp_bi200_ss100_s4_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

cd "$HOME/Reduce-then-optimize" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
chmod +x run_sil_training_bi200_ss100_class1_requestpruner.sh
./run_sil_training_bi200_ss100_class1_requestpruner.sh 4
