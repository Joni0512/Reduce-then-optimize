#!/bin/bash
# 2026-10-09: 4 additional seeds (101-104) of the best config so far - run E = wandb ethereal-sweep-1 (actor_lr 0.009315365931220558,
# critic_lr 0.0016399458556079478, 20 epochs; original seed 2079: best val 0.7389, test 0.7740) - as 4 parallel cm4_tiny jobs, see chat.
# cm4_tiny allows 4 jobs per user: jobs queue behind other running jobs. Outputs: outputs/srl_training_sweep/bestE_s<seed>/ (result.json).
set -eu
cd "$(dirname "$0")"
for seed in 101 102 103 104; do
  sbatch --job-name=srl_bestE --export=ALL,ACTOR_LR=0.009315365931220558,CRITIC_LR=0.0016399458556079478,SEED=$seed,EPOCHS=20,SRL_RUN_ID=bestE_s$seed submit_srl_buffered_actor_lr_trial.sbatch
done
