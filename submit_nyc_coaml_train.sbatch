#!/bin/bash
#SBATCH --job-name=nyc_coaml_train
#SBATCH --partition=teramem_inter
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_nyc_coaml_train_%j.out
#SBATCH --error=slurm_nyc_coaml_train_%j.err

set -e
cd "$SLURM_SUBMIT_DIR"
module load gurobi/13.00
source .venv/bin/activate

# 2026-07-19: proper COAML/SIL trip-scoring model training - trains the ScoringMLP on the
# 1100-request train pool across many epochs, evaluates (mode="eval", the model's own
# learned scores, no gradient) on the disjoint 300-request val pool each epoch, and keeps
# the checkpoint with the best val service rate. Fixes the earlier setup where every NYC
# "SIL" run trained a fresh, randomly-initialized model on the exact holdout it was then
# scored on, mode="train" only (expert-imitation replay, never the model's own policy).
# See rtv_solver/training_loop.py COAMLTrainingLoop._run_train_val_payloads.
#
# Uses the SAME pair-pruning GNN checkpoint (trained locally on the analogous disjoint
# 1100/300 split) that the final eval + RHO-ohne-Learning baseline runs below also use,
# so pruning is held constant and only the assignment policy differs between runs.
python3 rtv_solver/main.py \
  --mode coaml \
  --input_dir "" \
  --input_file solutions/nyc/manifests/nyc_train1100_imitation705_solve.json \
  --imitation_solution_file solutions/nyc/manifests/nyc_train1100_imitation705_solve.json \
  --val_input_file solutions/nyc/manifests/nyc_val300_imitation186_solve.json \
  --use_request_graph_pruner True \
  --request_graph_model_path outputs/models_nyc/rgnn_nyc_dense2000_full_train1100_val300_naive_pw1_v2/rgnn_nyc_dense2000_full_train1100_val300_naive_pw1_v2_best_val_f3.pt \
  --request_graph_threshold 0.5 \
  --max_cardinality 2 \
  --batch_interval 400 --step_size 100 \
  --epochs 20 \
  --learning_rate 0.0003 \
  --num_samples 20 \
  --seed 42 \
  --output_dir outputs/nyc_coaml_scoring_trained

echo "DONE nyc_coaml_train"
