#!/bin/bash
# 2026-10-22: episodic scheme with the "collect everything first" pretrain: ALL 38 training instances are played once with the random actor (buffer), then 380 critic steps (batch 20) from the buffer;
# main phase: per episode 1 actor + 1 critic step (batch 20), shared FIFO buffer over the last 380 episodes (= 10 epochs, filled step by step from the first episode on). Reward local_served, 1-step TD,
# actor_lr 0.0031923396291543868, critic_lr 0.002, 20 epochs, no lr annealing. Seed 301 runs locally (episodic_allpre_buf380_D_s301); this script submits seeds 302-304 to cm4_tiny
# (4-job limit, the sweep coordinator holds one slot; 64 G is plenty for the 380-episode compressed actor buffer). ~14-17 h expected (24 h limit).
# Output: outputs/srl_training_sweep/episodic_allpre_buf380_D_s<seed>/
set -eu
cd "$(dirname "$0")"
for seed in 302 303 304; do
  sbatch --clusters=cm4 --job-name=srl_allpre --export=ALL,ACTOR_LR=0.0031923396291543868,CRITIC_LR=0.002,SEED=$seed,EPOCHS=20,SRL_REWARD_MODE=local_served,SRL_ACTOR_UPDATE_MODE=episodic,SRL_PRETRAIN_MODE=buffer,SRL_PRETRAIN_INSTANCES=all,SRL_PRETRAIN_STEPS=380,SRL_EPISODIC_BUFFER_EPISODES=380,SRL_RUN_ID=episodic_allpre_buf380_D_s$seed submit_srl_buffered_actor_lr_trial.sbatch
done
