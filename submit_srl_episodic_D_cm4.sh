#!/bin/bash
# 2026-10-18: episodic scheme (same settings as submit_srl_episodic_D_cluster.sh: buffer pretrain 10 instances x 150 steps x batch 20, per episode 1 actor + 1 critic step, batch 20,
# shared buffer of the last 10 episodes, Polyak 0.005, reward local_served, config D, 20 epochs) on cm4_tiny for seeds 303-305, because serial_long (4 nodes, memory QOS) kept them pending.
# Seed 302 runs on serial_long (job 5671417), seed 301 locally. cm4_tiny: 24 h limit, 20 CPUs, 64 G (submit_srl_buffered_actor_lr_trial.sbatch); expected ~19-22 h
# (old srl_buf jobs took 16.7-20.8 h on cm4_tiny for a ~4.5 h local run, the episodic run needed 4.75 h locally) - tight, a timeout is possible (partial_val_curve.json is written after every validation).
# cm4_tiny allows 4 jobs per user at once (see CLAUDE.md, 2026-10-04); the sweep coordinator uses one slot, so 3 fit. Output: outputs/srl_training_sweep/episodic_D_s<seed>/ .
set -eu
cd "$(dirname "$0")"
for seed in 303 304 305; do
  sbatch --clusters=cm4 --job-name=srl_epi_D --export=ALL,ACTOR_LR=0.0031923396291543868,CRITIC_LR=0.002,SEED=$seed,EPOCHS=20,SRL_RUN_ID=episodic_D_s$seed,SRL_REWARD_MODE=local_served,SRL_ACTOR_UPDATE_MODE=episodic,SRL_PRETRAIN_MODE=buffer submit_srl_buffered_actor_lr_trial.sbatch
done
