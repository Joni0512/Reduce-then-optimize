#!/bin/bash
#SBATCH --job-name=sil_gnn_l2
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --qos=cm4_serial
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=sil_gnn_l2_%j.log
#SBATCH --error=sil_gnn_l2_%j.err

# 2026-09-08: GNN-actor SIL training on the mixed balanced split (gcn, 2
# layers) - see chat. Moved to the cluster after the local run died twice
# (session end / Mac sleep killed the background process despite nohup).
# Same hyperparameters as the existing MLP-actor mixed_balanced run, only
# model_type switches to gnn - see
# run_sil_training_bi200_ss100_mixed_balanced_legacy_gnn_gcn_l2_seed1.sh
# for the full rationale.
#
# python3 -u (unbuffered) - Python fully buffers stdout when piped, so
# print()/log output can sit unflushed for hours and be lost entirely if
# the job gets killed before flushing.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
./venv/bin/python3 -u rtv_solver/main.py \
  --mode coaml \
  --input_dir "solutions/li_lim/manifests/" \
  --override_training_files "lc101,lc102,lr101,lr102,lrc101,lrc102,lc201,lc202,lr201,lr202,lrc202,lrc203" \
  --override_validation_files "lc107,lc108,lr111,lr112,lrc107,lrc108,lc207,lc208,lr203,lr210,lrc201,lrc207" \
  --batch_interval 200 --step_size 100 --max_cardinality 2 \
  --learning_rate 0.0001 --epochs 5 \
  --imitation_scoring_rule legacy \
  --use_request_pruner False --use_request_graph_pruner False \
  --model_type gnn --gnn_num_message_passing_layers 2 --gnn_aggregator gcn \
  --seed 1 \
  --output_dir "outputs/sil_training_bi200_ss100_mixed_balanced_legacy_gnn_gcn_l2_seed1" \
  > sil_training_bi200_ss100_mixed_balanced_legacy_gnn_gcn_l2_seed1.log 2>&1
