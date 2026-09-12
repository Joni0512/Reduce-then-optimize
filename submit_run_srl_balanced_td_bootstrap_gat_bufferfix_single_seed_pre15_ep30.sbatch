#!/bin/bash
#SBATCH --job-name=srl_tdboot_gat_bufferfix_pre15ep30
#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --qos=cm4_serial_long
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=srl_tdboot_gat_bufferfix_pre15ep30_%j.log
#SBATCH --error=srl_tdboot_gat_bufferfix_pre15ep30_%j.err

# 2026-09-12: same as submit_run_srl_balanced_td_bootstrap_gat_bufferfix_single_seed
# .sbatch, but for the pre15_ep30 variant (15 critic-pretrain epochs, 30
# normal-training episodes, up from 10/20) - see chat. time raised from 12h
# to 18h since total episodes go from 30 to 45 (~50% more).
#
# Usage: sbatch --export=SEED=42 submit_run_srl_balanced_td_bootstrap_gat_bufferfix_single_seed_pre15_ep30.sbatch
#
# Prerequisite: outputs/outputs/sil_training_bi200_ss100_mixed_balanced_legacy_mlp_seed1/
# (the SIL actor checkpoint) must already exist on the cluster.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"

if [ -z "$SEED" ]; then
    echo "SEED not set - pass it via: sbatch --export=SEED=42 $0"
    exit 1
fi

./venv/bin/python3 -u -m rtv_solver.pipeline.run_srl_balanced_td_bootstrap_gat_bufferfix_single_seed_pre15_ep30 "$SEED" 2>&1 | grep --line-buffered -E "^===|^!!!|epoch [0-9]+ done" > "filtered_run_srl_balanced_td_bootstrap_gat_bufferfix_pre15ep30_seed${SEED}.log"
