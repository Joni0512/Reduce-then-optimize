#!/bin/bash
#SBATCH --job-name=coaml_expprefix_c3
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --qos=cm4_serial
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_coaml_expprefix_c3_%j.out
#SBATCH --error=slurm_coaml_expprefix_c3_%j.err

# 12-instance coaml_exponential_prefix baseline (no pruner), cardinality 3 --
# cluster counterpart of the local cardinality-2 batch in outputs/outputs/coaml_exponential_prefix/.

set -e
cd "$SLURM_SUBMIT_DIR"
chmod +x run_baseline_12_c3_exp_prefix.sh
./run_baseline_12_c3_exp_prefix.sh
