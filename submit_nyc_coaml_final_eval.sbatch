#!/bin/bash
#SBATCH --job-name=nyc_coaml_final_eval
#SBATCH --partition=teramem_inter
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_nyc_coaml_final_eval_%j.out
#SBATCH --error=slurm_nyc_coaml_final_eval_%j.err

set -e
cd "$SLURM_SUBMIT_DIR"
module load gurobi/13.00
source .venv/bin/activate

# 2026-07-19: the actual fair SIL comparison run - loads the checkpoint trained by
# submit_nyc_coaml_train.sbatch and solves the untouched 600-holdout-derived 200-request
# imitation manifest with mode="eval" (the model's own learned scores), not mode="train"
# (expert-imitation replay, what every earlier NYC SIL number reported). Same pruning
# checkpoint + threshold=0.5 as the RHO-ohne-Learning baseline run below, for a like-for-
# like comparison.
#
# IMPORTANT: only run this AFTER submit_nyc_coaml_train.sbatch has finished and
# outputs/nyc_coaml_scoring_trained/coaml_model_weights_best_val.pt exists.
python3 rtv_solver/main.py \
  --mode coaml \
  --input_dir "" \
  --input_file solutions/nyc/manifests/nyc_holdout_full_imitation200_solve.json \
  --imitation_solution_file solutions/nyc/manifests/nyc_holdout_full_imitation200_solve.json \
  --epochs 1 \
  --coaml_model_weights outputs/nyc_coaml_scoring_trained/coaml_model_weights_best_val.pt \
  --coaml_solve_mode eval \
  --use_request_graph_pruner True \
  --request_graph_model_path outputs/models_nyc/rgnn_nyc_dense2000_full_train1100_val300_naive_pw1_v2/rgnn_nyc_dense2000_full_train1100_val300_naive_pw1_v2_best_val_f3.pt \
  --request_graph_threshold 0.5 \
  --max_cardinality 2 \
  --batch_interval 400 --step_size 100 \
  --seed 42 \
  --output_dir outputs/nyc_sil_eval_final_t05

echo "DONE nyc_coaml_final_eval"
