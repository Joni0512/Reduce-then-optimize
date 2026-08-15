#!/bin/bash
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --qos=cm4_serial
#SBATCH --job-name=rh_c2_rp_bi400_ss80_s2
#SBATCH --output=rh_c2_rp_bi400_ss80_s2_%j.log
#SBATCH --error=rh_c2_rp_bi400_ss80_s2_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

cd "$HOME/Reduce-then-optimize" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
chmod +x run_rh_bi400_ss80_class2only_requestpruner.sh
./run_rh_bi400_ss80_class2only_requestpruner.sh 2
