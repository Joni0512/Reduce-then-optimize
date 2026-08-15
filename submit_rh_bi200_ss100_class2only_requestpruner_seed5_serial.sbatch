#!/bin/bash
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --qos=cm4_serial
#SBATCH --job-name=rh_c2_rp_bi200_ss100_s5
#SBATCH --output=rh_c2_rp_bi200_ss100_s5_%j.log
#SBATCH --error=rh_c2_rp_bi200_ss100_s5_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

cd "$HOME/Reduce-then-optimize" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
chmod +x run_rh_bi200_ss100_class2only_requestpruner.sh
./run_rh_bi200_ss100_class2only_requestpruner.sh 5
