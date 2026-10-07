#!/bin/bash
#SBATCH --job-name=srl_buf_ser
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --qos=cm4_serial
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=srl_buf_ser_%j.log
#SBATCH --error=srl_buf_ser_%j.err

# 2026-10-07: serial-cluster variant of submit_srl_buffered_actor_lr_trial.sbatch, to measure how long
# one buffered run takes on serial_std (cm4_tiny: 20.6-20.8 h for 10+20 epochs, local Mac: 4.3 h) - see chat.
# serial_std / cm4_serial (checked 2026-10-04): 24 h, cpu=192 per user, no memory limit; 16 G is generous
# (measured peak RSS of the local buffered runs: 4.0-4.2 GB). ACTOR_LR (required), SEED (default 42),
# EPOCHS (default 20) and CRITIC_LR (optional, default = baseline critic_lr of the trial script) are
# exported by the submitter. NOTE: cluster must be given with -M serial for squeue/sacct.
# Submit: sbatch --export=ALL,ACTOR_LR=0.0031923396291543868,SEED=42,EPOCHS=20 submit_srl_buffered_actor_lr_trial_serial.sbatch

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"

./venv/bin/python3 -u -m rtv_solver.pipeline.srl_buffered_actor_lr_trial "$ACTOR_LR" "${SEED:-42}" "${EPOCHS:-20}" ${CRITIC_LR:+"$CRITIC_LR"}
