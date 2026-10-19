#!/bin/bash
# 2026-10-19: 30-epoch runs of the episodic scheme (buffer pretrain 10 instances x 150 x batch 20, 1 actor + 1 critic step per episode, buffer 10 episodes, reward local_served, actor_lr 0.0031923396291543868,
# critic_lr 0.002): 2-step TD seeds 301 (local, episodic_td2_D_ep30_s301) + 302, and 1-step TD seeds 301 + 302. 1-step seed 301 runs locally after the td2 run. This script submits the two cluster runs: td2 s302 and 1-step s302.
# cm4_tiny has a 24 h limit and ~40-43 min per episodic epoch -> 30 epochs ~20-22 h + test: tight; partial_val_curve.json is written after every validation (epoch 5, 10, ...) if a job times out.
# Output: outputs/srl_training_sweep/episodic_{td2_,}D_ep30_s<seed>/
set -eu
cd "$(dirname "$0")"
COMMON=ACTOR_LR=0.0031923396291543868,CRITIC_LR=0.002,EPOCHS=30,SRL_REWARD_MODE=local_served,SRL_ACTOR_UPDATE_MODE=episodic,SRL_PRETRAIN_MODE=buffer
sbatch --clusters=cm4 --job-name=srl_ep30 --export=ALL,$COMMON,SRL_TD_STEPS=2,SEED=302,SRL_RUN_ID=episodic_td2_D_ep30_s302 submit_srl_buffered_actor_lr_trial.sbatch
for seed in 302; do  # 1-step seed 301 runs locally (episodic_D_ep30_s301, after the local td2 run)
  sbatch --clusters=cm4 --job-name=srl_ep30 --export=ALL,$COMMON,SEED=$seed,SRL_RUN_ID=episodic_D_ep30_s$seed submit_srl_buffered_actor_lr_trial.sbatch
done
