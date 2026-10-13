#!/bin/bash
# 2026-10-13: GNN actor (gcn, 2 layers, hidden 64) with config D (actor_lr 0.0031923396291543868, critic_lr 0.002), buffered actor update, 20 epochs, seeds 301-305
# (same seeds/parameters as the MLP runs bestD_s301-305) on serial_long (4 CPUs, 16 G each like the running GNN job 5656875; the GNN run died locally with a 24 GB peak on a 16 GB Mac).
# Output: outputs/srl_training_sweep/gnnactor_D_s<seed>/ . The serial_long QOS caps memory per user (~99.7 GB), so part of the jobs may wait.
set -eu
cd "$(dirname "$0")"
for seed in 301 302 303 304 305; do
  sbatch --clusters=serial --job-name=srl_gnn_D --export=ALL,ACTOR_LR=0.0031923396291543868,CRITIC_LR=0.002,SEED=$seed,EPOCHS=20,SRL_RUN_ID=gnnactor_D_s$seed,SRL_ACTOR_MODEL_TYPE=gnn,SRL_ACTOR_GNN_LAYERS=2,SRL_ACTOR_GNN_AGGREGATOR=gcn submit_srl_buffered_actor_lr_trial_serial_long.sbatch
done
