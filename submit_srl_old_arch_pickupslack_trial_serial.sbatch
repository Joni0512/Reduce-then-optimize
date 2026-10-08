#!/bin/bash
#SBATCH --job-name=srl_old_ser
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --qos=cm4_serial
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --output=srl_old_ser_%j.log
#SBATCH --error=srl_old_ser_%j.err

# 2026-10-08: replication runs of the OLD per-window actor architecture (twin critic, no warmstart, pickup_slack,
# actor_lr baseline, 10 pretrain + EPOCHS main epochs) on serial_std - one run per job. Purpose: test whether the old
# method's collapse at critic_lr 0.001/0.0015 (one run each so far) is systematic, to compare with the buffered
# method (see chat). Same trial script as the cm4 baseline sweep (srl_cluster_twin_critic_no_warmstart_pickupslack_cm4_trial.py,
# run_id cm4_ep<EPOCHS>_ps_clr<critic_lr>_s<seed> even when run on serial, so outputs land next to the earlier ones).
# 24 G: observed peak RSS of old-architecture runs up to ~18 G. serial_std / cm4_serial: 24 h, cpu=192 per user, no memory limit.
# Old architecture took ~13.5 h on cm4_tiny and ~19-21 h on serial_std for 20 epochs (earlier runs, not re-measured here).
# CRITIC_LR and SEED required, EPOCHS default 20, exported by the submitter (submit_srl_old_arch_replication_serial.sh).

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"

./venv/bin/python3 -u -m rtv_solver.pipeline.srl_cluster_twin_critic_no_warmstart_pickupslack_cm4_trial "$CRITIC_LR" "$SEED" "${EPOCHS:-20}"
