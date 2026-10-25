#!/bin/bash
# 2026-10-26: same as submit_srl_episodic_allpre_buf380_cluster.sh (all 38 instances collected once, 380 critic pretrain steps, buffer of 380 episodes, 1 actor + 1 critic step per episode, local_served, 1-step TD,
# actor_lr 0.0031923396291543868, 20 epochs) but critic_lr 0.004 (instead of 0.002). Seed 301 runs locally (episodic_allpre_buf380_clr0p004_s301); this script submits seeds 302-305:
# 302-304 -> cm4_tiny (4-job limit, the sweep coordinator holds one slot), 305 -> cm4_std (whole nodes, 112 CPUs requested). ~14-20 h expected (24 h limit).
# Output: outputs/srl_training_sweep/episodic_allpre_buf380_clr0p004_s<seed>/
set -eu
cd "$(dirname "$0")"
COMMON=ACTOR_LR=0.0031923396291543868,CRITIC_LR=0.004,EPOCHS=20,SRL_REWARD_MODE=local_served,SRL_ACTOR_UPDATE_MODE=episodic,SRL_PRETRAIN_MODE=buffer,SRL_PRETRAIN_INSTANCES=all,SRL_PRETRAIN_STEPS=380,SRL_EPISODIC_BUFFER_EPISODES=380
for seed in 302 303 304; do
  sbatch --clusters=cm4 --job-name=srl_clr004 --export=ALL,$COMMON,SEED=$seed,SRL_RUN_ID=episodic_allpre_buf380_clr0p004_s$seed submit_srl_buffered_actor_lr_trial.sbatch
done
sbatch --clusters=cm4 --partition=cm4_std --qos=cm4_std --cpus-per-task=112 --mem=64G --job-name=srl_clr004 --export=ALL,$COMMON,SEED=305,SRL_RUN_ID=episodic_allpre_buf380_clr0p004_s305 submit_srl_buffered_actor_lr_trial.sbatch
