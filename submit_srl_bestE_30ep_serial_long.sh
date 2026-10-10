#!/bin/bash
# 2026-10-10: 5 new seeds (201-205) of config E (actor_lr 0.009315365931220558, critic_lr 0.0016399458556079478) with 30 main epochs on
# serial_long, to test whether 30 epochs beat 20: every run's val curve contains the epoch-20 value (no LR schedule), so val @20 vs val @30 is a
# within-run comparison. Outputs: outputs/srl_training_sweep/bestE30_s<seed>/result.json. See chat.
set -eu
cd "$(dirname "$0")"
for seed in 201 202 203 204 205; do
  sbatch --job-name=srl_30ep --export=ALL,ACTOR_LR=0.009315365931220558,CRITIC_LR=0.0016399458556079478,SEED=$seed,EPOCHS=30,SRL_RUN_ID=bestE30_s$seed submit_srl_buffered_actor_lr_trial_serial_long.sbatch
done
