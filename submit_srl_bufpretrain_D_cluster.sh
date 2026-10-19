#!/bin/bash
# 2026-10-19: "buffered main phase + buffer pretrain" (user request, see chat): the OLD main phase (SRL_ACTOR_UPDATE_MODE=buffered: actor update once per epoch, batch 16, critic replay buffer 40) with the NEW
# pretrain (SRL_PRETRAIN_MODE=buffer: 10 instances x 150 steps x batch 20), reward local_positive and everything else as the 5 old D seeds (bestD_s301-305: config D, critic_lr 0.002, 1-step TD, 20 epochs).
# Purpose: separate the two changes made at once for the episodic runs - if val@20 is ~0.71 again, the short pretrain is innocent and the drop comes from the episodic main phase; if it drops too, the pretrain is the cause.
# Seed 301 runs locally (bufpre_D_s301). Seeds 302-304 -> cm4_tiny (4-job limit, coordinator holds one slot), seed 305 -> serial_long (16 G; ~22 h expected there, 3-day limit; 16 G fits next to the 4x20 G running jobs, ~99.7 GB limit).
# Output: outputs/srl_training_sweep/bufpre_D_s<seed>/
set -eu
cd "$(dirname "$0")"
COMMON=ACTOR_LR=0.0031923396291543868,CRITIC_LR=0.002,EPOCHS=20,SRL_REWARD_MODE=local_positive,SRL_PRETRAIN_MODE=buffer
for seed in 302 303 304; do
  sbatch --clusters=cm4 --job-name=srl_bufpre_D --export=ALL,$COMMON,SEED=$seed,SRL_RUN_ID=bufpre_D_s$seed submit_srl_buffered_actor_lr_trial.sbatch
done
sbatch --clusters=serial --job-name=srl_bufpre_D --mem=16G --export=ALL,$COMMON,SEED=305,SRL_RUN_ID=bufpre_D_s305 submit_srl_buffered_actor_lr_trial_serial_long.sbatch
