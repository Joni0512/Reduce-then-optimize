#!/bin/bash
# 2026-10-19: learning-rate grid for the episodic scheme (1-step TD, local_served, buffer pretrain 10 instances x 150 x batch 20, per episode 1 actor + 1 critic step, buffer 10 episodes, 20 epochs):
# actor_lr {0.0032 (base), 0.0016, 0.0008} x critic_lr {0.002 (base), 0.004}, seeds 301 and 302 (the same seeds as episodic_D_s301/302, the base cell itself already exists: episodic_D_s301-305).
# => 5 new cells x 2 seeds = 10 runs, spread over the free slots: cm4_tiny (limit 4 jobs per user, the sweep coordinator holds one -> 3 run, the rest pends), cm4_std (limit 2, 112+ CPUs / whole nodes),
# serial_long (16 G, memory QOS ~99.7 GB). Jobs above a limit simply pend (QOSMaxJobsPerUserLimit / QOSMaxMemoryPerUser) and start when a slot frees. Expected ~14-17 h per run on cm4 (~22 h on serial_long).
# Output: outputs/srl_training_sweep/epigrid_a<actor_lr>_c<critic_lr>_s<seed>/ . Rating: mean of the validations at epochs 10/15/20 (not best val).
set -eu
cd "$(dirname "$0")"
COMMON=EPOCHS=20,SRL_REWARD_MODE=local_served,SRL_ACTOR_UPDATE_MODE=episodic,SRL_PRETRAIN_MODE=buffer
tag() { echo "$1" | tr '.' 'p'; }
submit_tiny() {  # actor_lr critic_lr seed
  sbatch --clusters=cm4 --job-name=srl_grid --export=ALL,$COMMON,ACTOR_LR=$1,CRITIC_LR=$2,SEED=$3,SRL_RUN_ID=epigrid_a$(tag $1)_c$(tag $2)_s$3 submit_srl_buffered_actor_lr_trial.sbatch
}
submit_std() {
  sbatch --clusters=cm4 --partition=cm4_std --qos=cm4_std --cpus-per-task=112 --mem=64G --job-name=srl_grid --export=ALL,$COMMON,ACTOR_LR=$1,CRITIC_LR=$2,SEED=$3,SRL_RUN_ID=epigrid_a$(tag $1)_c$(tag $2)_s$3 submit_srl_buffered_actor_lr_trial.sbatch
}
submit_long() {
  sbatch --clusters=serial --job-name=srl_grid --mem=16G --export=ALL,$COMMON,ACTOR_LR=$1,CRITIC_LR=$2,SEED=$3,SRL_RUN_ID=epigrid_a$(tag $1)_c$(tag $2)_s$3 submit_srl_buffered_actor_lr_trial_serial_long.sbatch
}
# cm4_tiny (5): the two most informative cells first
submit_tiny 0.0016 0.002 301; submit_tiny 0.0016 0.002 302
submit_tiny 0.0016 0.004 301; submit_tiny 0.0016 0.004 302
submit_tiny 0.0032 0.004 301
# cm4_std (3)
submit_std 0.0032 0.004 302
submit_std 0.0008 0.002 301; submit_std 0.0008 0.002 302
# serial_long (2)
submit_long 0.0008 0.004 301; submit_long 0.0008 0.004 302
