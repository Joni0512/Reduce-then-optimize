#!/bin/bash
# 2026-10-26: SRL at "half horizon": batch_interval=60, step_size=30 (RHO reference: batch_interval=120, step_size=30). Pretrain: 10 epochs collected with exploration noise (sigma 1.0; sigma 0.2 gave nearly
# identical epochs), all 38 instances per epoch -> 380 different episodes in the buffer, then 2000 critic steps (batch 20). Main phase: 1 actor + 1 critic step per episode (batch 20), FIFO buffer of 380 episodes,
# reward local_served, 1-step TD, actor_lr 0.0031923396291543868, critic_lr 0.002, 20 epochs. Seed 301 runs locally (hh_bi60ss30_allpre10ep_s301); this script submits seeds 302-305.
# Runtime: the smaller window roughly doubles the number of windows per episode -> expect ~2x the 200/100 time (~80 min/epoch on cm4_tiny = 27+ h, which does NOT fit the 24 h of cm4_tiny/cm4_std),
# so the jobs go to serial_long (up to 7 days; ~2 h/epoch there = ~45 h). Memory 24 G (buffer of ~7800 compressed windows); serial_long's memory QOS (~99.7 GB) and the queue of other jobs may delay the start.
# Output: outputs/srl_training_sweep/hh_bi60ss30_allpre10ep_s<seed>/
set -eu
cd "$(dirname "$0")"
for seed in 302 303 304 305; do
  sbatch --clusters=serial --job-name=srl_hh --mem=24G --time=5-00:00:00 --export=ALL,ACTOR_LR=0.0031923396291543868,CRITIC_LR=0.002,SEED=$seed,EPOCHS=20,SRL_BATCH_INTERVAL=60,SRL_STEP_SIZE=30,SRL_PRETRAIN_COLLECT_EPOCHS=10,SRL_PRETRAIN_SIGMA=1.0,SRL_PRETRAIN_STEPS=2000,SRL_PRETRAIN_INSTANCES=all,SRL_EPISODIC_BUFFER_EPISODES=380,SRL_REWARD_MODE=local_served,SRL_ACTOR_UPDATE_MODE=episodic,SRL_PRETRAIN_MODE=buffer,SRL_RUN_ID=hh_bi60ss30_allpre10ep_s$seed submit_srl_buffered_actor_lr_trial_serial_long.sbatch
done
