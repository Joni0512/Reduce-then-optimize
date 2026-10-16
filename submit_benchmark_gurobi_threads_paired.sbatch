#!/bin/bash
#SBATCH --job-name=bench_thr_pair
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --qos=cm4_serial
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=8G
#SBATCH --output=bench_thr_pair_%j.log
#SBATCH --error=bench_thr_pair_%j.err

# 2026-10-17: paired Gurobi-thread benchmark (LRZ servicedesk incident 269588): ALL variants run one after another INSIDE ONE JOB, i.e. on the same node, same 4 cores and the same
# surroundings, so only the thread setting differs (the earlier single jobs landed on different nodes and ran side by side). RHO on the 9 validation instances
# (batch_interval=200, step_size=100, cardinality 2, keep_active=false), order GUROBI_THREADS = 1, 4, unset, 1 (the last run repeats the first to see drift over time).
# Each variant uses its own seed (4301..4304) only to get its own output folder in outputs/rho_val_runs, so the RHO reference runs (seed 42) are not touched.
# Summary lines "VARIANT ... total ...s" at the end of the log. Submit: sbatch submit_benchmark_gurobi_threads_paired.sbatch

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
export SRL_MAX_THREAD_CNT=4
echo "node=$(hostname) cpus=$SLURM_CPUS_PER_TASK $(lscpu | grep -m1 'Model name')"
run_variant() {  # name, seed, GUROBI_THREADS value ("" = unset)
  local start=$(date +%s)
  if [ -n "$3" ]; then
    GUROBI_THREADS=$3 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK \
      ./venv/bin/python3 -u -m rtv_solver.pipeline.rho_val_horizon_runs --batch_interval 200 --step_size 100 --cardinality 2 --seed $2 --keep_active false --runs 1 > variant_$1.out 2>&1
  else
    ./venv/bin/python3 -u -m rtv_solver.pipeline.rho_val_horizon_runs --batch_interval 200 --step_size 100 --cardinality 2 --seed $2 --keep_active false --runs 1 > variant_$1.out 2>&1
  fi
  echo "VARIANT $1 (GUROBI_THREADS=${3:-unset}): total $(( $(date +%s) - start )) s | $(grep -a 'MEAN over' variant_$1.out | sed 's/.*MEAN over/MEAN over/')"
  grep -a "lc202:\|lr202:\|lr208:" variant_$1.out | sed 's/.*run=1\] /    /'
}
run_variant thr1_a 4301 1
run_variant thr4 4302 4
run_variant default 4303 ""
run_variant thr1_b 4304 1
