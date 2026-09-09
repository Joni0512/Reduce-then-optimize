#!/bin/bash
#SBATCH --job-name=srl_sweep_tdboot
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --qos=cm4_serial
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=srl_sweep_tdboot_%j.log
#SBATCH --error=srl_sweep_tdboot_%j.err

# 2026-09-09: wandb sweep over TD-bootstrap's gamma (0.95-0.99) and Polyak
# target_critic tau (0.001-0.005), at bi200/ss100 - see chat and
# sweep_td_bootstrap_trial.py's own docstring. Checks whether some
# combination in this range avoids the lr/lrc-instance collapse seen at
# gamma=0.99/tau=0.005.
#
# Prerequisite: outputs/srl_sweep/shared_critic_pretrained.pt (built via
# pretrain_and_save_shared_critic.py) AND the SIL actor checkpoint under
# outputs/outputs/sil_training_bi200_ss100_mixed_balanced_legacy_mlp_seed1/
# must already exist on the cluster.
#
# 2026-09-04 wandb-core fix (see chat, submit_srl_replaybuffer_sweep_v3.sbatch):
# baked-in WANDB__SERVICE_WAIT + node-local WANDB_DIR to avoid the recurring
# "the service process is busy and did not respond in time" startup failure.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
export WANDB__SERVICE_WAIT=600
export WANDB_DIR="/tmp/wandb_${SLURM_JOB_ID}"
mkdir -p "$WANDB_DIR"

SWEEP_ID_FILE="sweep_td_bootstrap_id.txt"
if [ ! -f "$SWEEP_ID_FILE" ]; then
    ./venv/bin/wandb sweep --project srl-td-bootstrap-sweep sweep_td_bootstrap.yaml 2>&1 | tee /tmp/sweep_tdboot_create_${SLURM_JOB_ID}.log
    SWEEP_ID=$(grep -oE "[a-z0-9]+/[a-z0-9]+/[a-z0-9]+$" /tmp/sweep_tdboot_create_${SLURM_JOB_ID}.log | tail -1)
    echo "$SWEEP_ID" > "$SWEEP_ID_FILE"
fi
SWEEP_ID=$(cat "$SWEEP_ID_FILE")
echo "Using sweep: $SWEEP_ID"

./venv/bin/wandb agent "$SWEEP_ID" 2>&1 | grep --line-buffered -E "^wandb:|Agent (starting|received)|final_service_rate|mean_final_service_rate" > filtered_sweep_td_bootstrap.log
