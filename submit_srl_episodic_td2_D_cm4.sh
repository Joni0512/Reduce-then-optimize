#!/bin/bash
# 2026-10-18: episodic scheme with the TWO-STEP TD target (SRL_TD_STEPS=2: y = r_t + g*r_{t+1} + g^2*Q'(s_t+2)), otherwise identical to submit_srl_episodic_D_cm4.sh (config D, local_served, buffer pretrain
# 10 instances x 150 x batch 20, per episode 1 actor + 1 critic step, buffer of the last 10 episodes). Compare with episodic_D_s<seed> (same seeds). Seed 301 runs locally (episodic_td2_D_s301).
# Seeds 302/303 go to cm4_std (2 free slots, cm4_tiny's 4 slots are used by the coordinator + episodic seeds 303-305). cm4_std needs MinTRES cpu=112 (whole-node jobs only, see
# submit_sil_mixedall_v1final_c3_t06.sbatch), so 112 CPUs are requested although the run only spawns 16 workers; 24 h limit, expected ~19-22 h (tight). Seed 304: add via
# `./submit_srl_episodic_td2_D_cm4.sh 304` once a slot is free. Command-line options override the #SBATCH lines of submit_srl_buffered_actor_lr_trial.sbatch.
set -eu
cd "$(dirname "$0")"
SEEDS=${@:-302 303}
for seed in $SEEDS; do
  sbatch --clusters=cm4 --partition=cm4_std --qos=cm4_std --cpus-per-task=112 --mem=64G --job-name=srl_epi_td2_D --export=ALL,ACTOR_LR=0.0031923396291543868,CRITIC_LR=0.002,SEED=$seed,EPOCHS=20,SRL_RUN_ID=episodic_td2_D_s$seed,SRL_REWARD_MODE=local_served,SRL_ACTOR_UPDATE_MODE=episodic,SRL_PRETRAIN_MODE=buffer,SRL_TD_STEPS=2 submit_srl_buffered_actor_lr_trial.sbatch
done
