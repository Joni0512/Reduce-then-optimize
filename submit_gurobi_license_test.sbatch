#!/bin/bash
#SBATCH --job-name=gurobi_license_test
#SBATCH --partition=teramem_inter
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=slurm_gurobi_license_test_%j.out
#SBATCH --error=slurm_gurobi_license_test_%j.err

set -e
cd "$SLURM_SUBMIT_DIR"
module load gurobi/13.00
source .venv/bin/activate
python3 test_gurobi_license.py
echo "DONE gurobi_license_test"
