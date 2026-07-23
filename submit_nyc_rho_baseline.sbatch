#!/bin/bash
#SBATCH --job-name=nyc_rho_baseline
#SBATCH --partition=teramem_inter
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_nyc_rho_baseline_%j.out
#SBATCH --error=slurm_nyc_rho_baseline_%j.err

set -e
cd "$SLURM_SUBMIT_DIR"
module load gurobi/13.00
source .venv/bin/activate

# 2026-07-19: RHO ohne Learning baseline, rerun with the NEW pair-pruning checkpoint
# (train1100/val300 disjoint split) so it's directly comparable to the SIL eval-mode run
# in submit_nyc_coaml_final_eval.sbatch - same checkpoint, same threshold=0.5, only the
# assignment policy differs (plain cost-minimization here vs. the learned score model there).
python3 rtv_solver/main.py \
  --mode offline \
  --input_file solutions/nyc/manifests/nyc_holdout300_solve.json \
  --use_request_graph_pruner True \
  --request_graph_model_path outputs/models_nyc/rgnn_nyc_dense2000_full_train1100_val300_naive_pw1_v2/rgnn_nyc_dense2000_full_train1100_val300_naive_pw1_v2_best_val_f3.pt \
  --request_graph_threshold 0.5 \
  --max_cardinality 2 \
  --batch_interval 400 --step_size 100 \
  --seed 42 \
  --output_dir outputs/nyc_rho_baseline_t05

echo "DONE nyc_rho_baseline"
