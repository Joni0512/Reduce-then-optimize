#!/bin/bash
#SBATCH --job-name=srl_gat_gnnact
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --qos=cm4_serial
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=srl_gat_gnnact_%j.log
#SBATCH --error=srl_gat_gnnact_%j.err

# 2026-09-08: first SRL+GAT test with a GNN actor (instead of MLP) - see
# chat. 1 seed, all 12 balanced instances, same GAT critic setup as
# replicate_gat_nocliqueue_12instances_seeds.py. Must run on the cluster -
# the GNN-actor checkpoint (submit_sil_training_mixed_balanced_gnn_gcn_l2_seed1.sbatch)
# only exists here, not synced locally.
#
# python3 -u (unbuffered) + grep --line-buffered - Python fully buffers
# stdout when piped, so print() output can sit unflushed for hours and be
# lost entirely if the job gets killed before flushing.
#
# Prerequisite: outputs/outputs/sil_training_bi200_ss100_mixed_balanced_legacy_gnn_gcn_l2_seed1/
# (the GNN actor checkpoint, from submit_sil_training_mixed_balanced_gnn_gcn_l2_seed1.sbatch)
# must already exist on the cluster.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
./venv/bin/python3 -u -m rtv_solver.pipeline.test_srl_gat_gnnactor_12instances 2>&1 | grep --line-buffered -E "^===|^!!!|epoch [0-9]+ done" > filtered_test_srl_gat_gnnactor_12instances.log
