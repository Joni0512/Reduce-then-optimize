#!/bin/bash
#SBATCH --job-name=srl_rb_sweep_v3
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --qos=cm4_serial
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=srl_rb_sweep_v3_%j.log
#SBATCH --error=srl_rb_sweep_v3_%j.err

# 2026-09-02: cluster agent for the v3 SRL replay-buffer wandb sweep (see
# chat and sweep_srl_replaybuffer_trial_v3.py's own docstring) - narrower,
# shifted-upward critic_lr/replay_capacity ranges based on v2's (3sxp7i3o)
# results. User's call: cluster-only for this sweep, not local. Target
# ~50 finished trials, same as v2.
#
# Prerequisite: outputs/srl_sweep/shared_critic_pretrained.pt (built via
# pretrain_and_save_shared_critic.py) AND the SIL actor checkpoint under
# outputs/outputs/sil_training_bi200_ss100_mixed_balanced_legacy_mlp_seed1/
# must already exist on the cluster - rsync both over before submitting.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
# 2026-09-02: filtered through grep instead of letting raw solver stdout/
# stderr flood the SBATCH --output/--error files - wandb agent streams each
# trial subprocess's full output through, same runaway-log risk as the other
# cluster jobs (one .err file grew to 17.5GB and exceeded the shared disk
# quota, killing every other concurrently running job too - see chat).
# 2026-09-03: grep --line-buffered - see chat (wandb's own CLI output is
# already line-buffered by default, only the python -u fix applies to our
# own scripts' direct stdout).
./venv/bin/wandb agent jonathan-homafar-technical-university-of-munich/srl-replaybuffer-sweep/05cd04mw 2>&1 | grep --line-buffered -E "^wandb:|Agent (starting|received)|final_service_rate|mean_final_service_rate" > filtered_srl_replaybuffer_sweep_v3.log
