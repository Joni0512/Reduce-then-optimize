#!/bin/bash
#SBATCH --job-name=srl_30ep
#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --qos=cm4_serial_long
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=srl_30ep_%j.log
#SBATCH --error=srl_30ep_%j.err

# 2026-10-10: buffered SRL trial on serial_long (7 days) for 30-epoch runs (serial_std's 24 h do not fit 10 pretrain + 30 epochs: ~28-31 h expected).
# 4 CPUs + SRL_MAX_THREAD_CNT=4: the CPU-scaling benchmark (serial_std) showed pool size = CPUs is best and 4 CPUs are enough (see chat).
# serial_long QOS cm4_serial_long: cpu=192 and mem~99.7 GB per user -> 16 G per job leaves room for ~5-6 such jobs next to other long jobs.
# ACTOR_LR, SEED, EPOCHS, CRITIC_LR (optional) and SRL_RUN_ID exported by the submitter.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
export SRL_MAX_THREAD_CNT=4
./venv/bin/python3 -u -m rtv_solver.pipeline.srl_buffered_actor_lr_trial "$ACTOR_LR" "${SEED:-42}" "${EPOCHS:-30}" ${CRITIC_LR:+"$CRITIC_LR"}
