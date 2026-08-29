#!/bin/bash
#SBATCH --job-name=srl_rb_sweep_long
#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --qos=cm4_serial_long
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=srl_rb_sweep_long_%j.log
#SBATCH --error=srl_rb_sweep_long_%j.err

# 2026-08-29: cluster agent for the SRL replay-buffer wandb sweep (see chat)
# - companion to submit_srl_replaybuffer_sweep_serial.sbatch (serial_std,
# 24h cap); this one runs on serial_long for a longer wall-clock budget.
# Both point at the SAME sweep-id, so they (and any local agent) split the
# work automatically - wandb's controller hands out the next hyperparameter
# combination to whichever agent asks next, no manual partitioning needed.
#
# Prerequisite: outputs/srl_sweep/shared_critic_pretrained.pt must already
# exist (built once locally via pretrain_and_save_shared_critic.py) - rsync
# or scp the whole outputs/srl_sweep/ dir (and the SIL actor checkpoint under
# outputs/outputs/sil_training_bi200_ss100_mixed_balanced_legacy_mlp_seed1/)
# to the cluster before submitting, since this job does NOT re-pretrain it.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
./venv/bin/wandb agent jonathan-homafar-technical-university-of-munich/srl-replaybuffer-sweep/m8wzikdz
