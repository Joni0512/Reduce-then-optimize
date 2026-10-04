#!/bin/bash
# Submits the 4 critic_lr x 2 seed (42, 43) = 8 baseline jobs on cm4 (20 epochs, pickup_slack,
# actor_lr baseline) - see chat 2026-10-04. Seed 42 -> cm4_tiny (4 jobs at once), seed 43 ->
# cm4_std (2 at once, the rest queue). Needs the QOS names to match the partition names
# (checked in sacctmgr: cm4_std / cm4_tiny exist); if sbatch rejects them, check
# `scontrol -M cm4 show partition cm4_tiny`.
set -eu
cd "$(dirname "$0")"
for clr in 0.0006 0.0008 0.0010833586558285635 0.0015; do
  sbatch --export=ALL,CRITIC_LR=$clr,SEED=42,EPOCHS=20 --partition=cm4_tiny --qos=cm4_tiny submit_srl_cluster_twin_critic_no_warmstart_pickupslack_cm4_trial.sbatch
  sbatch --export=ALL,CRITIC_LR=$clr,SEED=43,EPOCHS=20 --partition=cm4_std --qos=cm4_std submit_srl_cluster_twin_critic_no_warmstart_pickupslack_cm4_trial.sbatch
done
