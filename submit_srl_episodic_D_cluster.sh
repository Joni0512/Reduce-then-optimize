#!/bin/bash
# 2026-10-17: episodic scheme (buffer pretrain: 10 instances x 150 steps x batch 20; main phase: per episode 1 actor + 1 critic step, batch 20, shared FIFO buffer of the last
# 10 episodes, Polyak 0.005 per step), reward local_served, config D (actor_lr 0.0031923396291543868, critic_lr 0.002), 20 epochs. Seed 301 runs locally (episodic_D_s301);
# this script submits seeds 302-305 to serial_long (expected ~25 h on the cluster = 5.8x the local ~4-5 h, i.e. too long for serial_std's 24 h). 4 CPUs, 20 G per job;
# serial_long's memory QOS (~99.7 GB per user) may keep part of the jobs pending. Output: outputs/srl_training_sweep/episodic_D_s<seed>/ .
set -eu
cd "$(dirname "$0")"
for seed in 302 303 304 305; do
  sbatch --clusters=serial --job-name=srl_epi_D --mem=20G --export=ALL,ACTOR_LR=0.0031923396291543868,CRITIC_LR=0.002,SEED=$seed,EPOCHS=20,SRL_RUN_ID=episodic_D_s$seed,SRL_REWARD_MODE=local_served,SRL_ACTOR_UPDATE_MODE=episodic,SRL_PRETRAIN_MODE=buffer submit_srl_buffered_actor_lr_trial_serial_long.sbatch
done
