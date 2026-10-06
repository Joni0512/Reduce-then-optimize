#!/bin/bash
# 2026-10-06: submits the 3 CLUSTER runs of the buffered actor_lr scan (2x/4x/8x the old baseline
# 0.0031923396291543868, seed 42, 20 epochs); the 1x run (0.0032) is run locally - see chat.
set -eu
cd "$(dirname "$0")"
for alr in 0.0064 0.0128 0.0255; do
  sbatch --export=ALL,ACTOR_LR=$alr,SEED=42,EPOCHS=20 submit_srl_buffered_actor_lr_trial.sbatch
done
