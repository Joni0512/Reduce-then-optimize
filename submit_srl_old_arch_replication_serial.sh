#!/bin/bash
# 2026-10-08: submits 5 replication runs of the OLD architecture on serial_std (20 epochs each): critic_lr 0.001 with
# seeds 42/43/44 and critic_lr 0.0015 with seeds 44/45 (0.0015 seeds 42/43 already exist from the cm4 sweep) - see chat.
set -eu
cd "$(dirname "$0")"
for pair in "0.001 42" "0.001 43" "0.001 44" "0.0015 44" "0.0015 45"; do
  set -- $pair
  sbatch --export=ALL,CRITIC_LR=$1,SEED=$2,EPOCHS=20 submit_srl_old_arch_pickupslack_trial_serial.sbatch
done
