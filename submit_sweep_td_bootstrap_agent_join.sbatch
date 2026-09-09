#!/bin/bash
#SBATCH --job-name=srl_sweep_tdboot_join
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --qos=cm4_serial
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=srl_sweep_tdboot_join_%j.log
#SBATCH --error=srl_sweep_tdboot_join_%j.err

# 2026-09-09: joins the ALREADY-CREATED local TD-bootstrap sweep
# (jonathan-homafar-technical-university-of-munich/srl-td-bootstrap-sweep/y7d7mmmc,
# created locally, target 50 trials total) as a second agent, instead of
# creating a separate sweep - see chat. Splits the 50-trial budget between
# the local machine and this cluster job so it finishes faster, without
# duplicating work. Also sidesteps the "service process is busy" wandb-core
# sweep-creation failure seen in submit_sweep_td_bootstrap.sbatch, since no
# `wandb sweep` create call happens here - just `wandb agent` on the
# existing sweep ID.
#
# Prerequisite: outputs/srl_sweep/shared_critic_pretrained.pt AND the SIL
# actor checkpoint under
# outputs/outputs/sil_training_bi200_ss100_mixed_balanced_legacy_mlp_seed1/
# must already exist on the cluster.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
export WANDB__SERVICE_WAIT=600
export WANDB_DIR="/tmp/wandb_${SLURM_JOB_ID}"
mkdir -p "$WANDB_DIR"

SWEEP_ID="jonathan-homafar-technical-university-of-munich/srl-td-bootstrap-sweep/y7d7mmmc"
echo "Joining existing sweep: $SWEEP_ID"
./venv/bin/wandb agent "$SWEEP_ID" 2>&1 | grep --line-buffered -E "^wandb:|Agent (starting|received)|final_service_rate|mean_final_service_rate" > filtered_sweep_td_bootstrap_join.log
