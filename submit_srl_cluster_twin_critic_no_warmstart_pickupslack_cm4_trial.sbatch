#!/bin/bash
#SBATCH --job-name=srl_cm4e20
#SBATCH --clusters=cm4
#SBATCH --partition=cm4_tiny
#SBATCH --qos=cm4_tiny
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --output=srl_cm4e20_%j.log
#SBATCH --error=srl_cm4e20_%j.err

# cm4 baseline protocol for the old per-step SRL architecture - see
# srl_cluster_twin_critic_no_warmstart_pickupslack_cm4_trial.py's docstring. cm4_tiny (checked via
# sacctmgr/scontrol 2026-10-04): MinTRES cpu=17 per job (so 16 CPUs was rejected with
# QOSMinCpuNotSatisfied), max 4 jobs per user, 1 node per job, 24 h, no CPU/memory limit per user.
# To get all 8 runs (4 critic_lr x seeds 42/43) into ONE wave, each job runs the seeds in SEEDS
# (colon-separated, default "42:43") as parallel processes on its node: 20 CPUs, 64 G (observed up to ~18 G
# MaxRSS per run on serial). CRITIC_LR (required), SEEDS and EPOCHS exported by the submitter.
# Expected runtime ~19-21 h at EPOCHS=20, so 30 epochs do NOT fit. Per-seed output goes to
# srl_cm4e20_<jobid>_s<seed>.out; if the job hits the 24 h limit, all its seeds are lost together.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"

seed_list="${SEEDS:-42:43}"
for seed in ${seed_list//:/ }; do
  ./venv/bin/python3 -u -m rtv_solver.pipeline.srl_cluster_twin_critic_no_warmstart_pickupslack_cm4_trial "$CRITIC_LR" "$seed" "${EPOCHS:-20}" > "srl_cm4e20_${SLURM_JOB_ID}_s${seed}.out" 2>&1 &
done
wait
