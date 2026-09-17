#!/bin/bash
#SBATCH --job-name=sil_lr_sweep
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --qos=cm4_serial
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=sil_lr_sweep_%j.log
#SBATCH --error=sil_lr_sweep_%j.err

# 2026-09-16: one actor_lr value of the plain-SIL-from-scratch actor_lr
# sweep (same seed=42, 5 epochs, batch_interval=400/step_size=100) requested
# in chat - see rtv_solver/pipeline/sil_lr_sweep_run.py's module docstring.
# ACTOR_LR and (optional) INSTANCE_SET (mixed/class1, default mixed) are
# passed via --export by the submitting shell loop.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"

if [ -z "$ACTOR_LR" ]; then
    echo "ACTOR_LR not set - submit with sbatch --export=ACTOR_LR=<value>[,INSTANCE_SET=mixed|class1] submit_sil_lr_sweep_run.sbatch" >&2
    exit 1
fi

./venv/bin/python3 -u -m rtv_solver.pipeline.sil_lr_sweep_run "$ACTOR_LR" "${INSTANCE_SET:-mixed}"
