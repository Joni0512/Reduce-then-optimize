#!/bin/bash
# Submits the 4 critic_lr jobs of the cm4 baseline sweep (20 epochs, pickup_slack, actor_lr
# baseline); each job runs seeds 42 and 43 as parallel processes (8 runs total, one wave on
# cm4_tiny, which allows 4 jobs per user and needs >= 17 CPUs per job) - see chat 2026-10-04 and
# the sbatch file's header.
set -eu
cd "$(dirname "$0")"
for clr in 0.0006 0.0008 0.0010833586558285635 0.0015; do
  sbatch --export=ALL,CRITIC_LR=$clr,SEEDS=42:43,EPOCHS=20 submit_srl_cluster_twin_critic_no_warmstart_pickupslack_cm4_trial.sbatch
done
