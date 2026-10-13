#!/bin/bash
#SBATCH --job-name=srl_abl
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --qos=cm4_serial
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=srl_abl_%j.log
#SBATCH --error=srl_abl_%j.err

# 2026-10-13: buffered SRL trial on serial_std (24 h, QOS cm4_serial: cpu=192 per user, no memory limit) with 4 CPUs + SRL_MAX_THREAD_CNT=4 (the CPU-scaling
# benchmark showed pool size = CPUs is best and 4 CPUs are enough, 4/4 about 8 % faster than 16/16). Used for the ablation sweep (submit_srl_ablation_sweep_cluster.sh).
# 20 epochs + 10 pretrain took 22.7-24 h with 16 CPUs on serial_std (borderline!), so a run can hit the 24 h limit; partial_val_curve.json is written after every
# validation, so the val curve up to the last validated epoch survives a timeout. ACTOR_LR, CRITIC_LR, SEED, EPOCHS, SRL_RUN_ID and the SRL_* ablation env vars come from the submitter.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
export SRL_MAX_THREAD_CNT=4
./venv/bin/python3 -u -m rtv_solver.pipeline.srl_buffered_actor_lr_trial "$ACTOR_LR" "${SEED:-42}" "${EPOCHS:-20}" ${CRITIC_LR:+"$CRITIC_LR"}
