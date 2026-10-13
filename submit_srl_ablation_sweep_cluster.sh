#!/bin/bash
# 2026-10-13: small ablation sweep of the fixed SRL hyperparameters (see chat): base = config D (actor_lr 0.0031923396291543868, critic_lr 0.002),
# buffered actor update, 20 epochs, seeds 301 and 302; per parameter two new values -> 8 configs x 2 seeds = 16 runs.
# Values: sigma 0.1 / 0.3, samples 10 / 40, Polyak rate 0.001 / 0.01, gamma 0.98 / 0.97. Output: outputs/srl_training_sweep/abl_<param><value>_s<seed>/.
# 14 runs go to serial_std (4 CPUs, 24 h limit, no memory limit; 20 epochs took 22.7-24 h with 16 CPUs there - borderline, see the sbatch file);
# the 2 runs with samples=40 (longer actor-update phase, expected to exceed 24 h) go to serial_long (3 days, counts against the ~99.7 GB memory QOS, so they may wait).
set -eu
cd "$(dirname "$0")"
ALR=0.0031923396291543868; CLR=0.002
submit_std() { # name, env assignments, seed
  sbatch --clusters=serial --job-name="abl_$1" --export=ALL,ACTOR_LR=$ALR,CRITIC_LR=$CLR,SEED=$3,EPOCHS=20,SRL_RUN_ID=abl_$1_s$3,$2 submit_srl_buffered_trial_serial_std_4cpu.sbatch
}
submit_long() {
  sbatch --clusters=serial --job-name="abl_$1" --export=ALL,ACTOR_LR=$ALR,CRITIC_LR=$CLR,SEED=$3,EPOCHS=20,SRL_RUN_ID=abl_$1_s$3,$2 submit_srl_buffered_actor_lr_trial_serial_long.sbatch
}
for seed in 301 302; do
  submit_std sigma0p1 SRL_SIGMA=0.1 $seed
  submit_std sigma0p3 SRL_SIGMA=0.3 $seed
  submit_std samples10 SRL_NUM_SAMPLES=10 $seed
  submit_long samples40 SRL_NUM_SAMPLES=40 $seed
  submit_std polyak0p001 SRL_POLYAK_TAU=0.001 $seed
  submit_std polyak0p01 SRL_POLYAK_TAU=0.01 $seed
  submit_std gamma0p98 SRL_GAMMA=0.98 $seed
  submit_std gamma0p97 SRL_GAMMA=0.97 $seed
done
