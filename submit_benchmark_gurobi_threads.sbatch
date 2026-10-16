#!/bin/bash
#SBATCH --job-name=bench_thr
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --qos=cm4_serial
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=bench_thr_%j.log
#SBATCH --error=bench_thr_%j.err

# 2026-10-16: timing test for the LRZ servicedesk hint (incident 269588): does limiting Gurobi's threads (and OpenMP/MKL threads) speed up our runs on the cluster?
# RHO on the 9 validation instances (batch_interval=200, step_size=100, cardinality 2, keep_active=False; seed 4242 only to get its own output folder
# outputs/rho_val_runs/..._seed4242 so the RHO reference runs are not touched); the per-instance wall times are printed in the log ("(12.3s)").
# Submit: sbatch --export=ALL,GUROBI_THREADS=1,TAG=thr1 submit_benchmark_gurobi_threads.sbatch   (no GUROBI_THREADS = default behavior)
#         sbatch --hint=nomultithread --export=ALL,GUROBI_THREADS=1,TAG=thr1_nomt submit_benchmark_gurobi_threads.sbatch
# 2026-10-16: --nodes=1 / --ntasks-per-node=1 written out explicitly (LRZ servicedesk suggestion); GUROBI_THREADS=4 (= cpus) is the 4th variant to try.
# Run each variant on the same kind of node, ideally not at the same time as many other jobs of yours (they share the node).

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
export SRL_MAX_THREAD_CNT=4
if [ -n "${GUROBI_THREADS:-}" ]; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi
echo "variant ${TAG:-default}: GUROBI_THREADS=${GUROBI_THREADS:-unset} cpus=$SLURM_CPUS_PER_TASK node=$(hostname) $(lscpu | grep -m1 'Model name')"
date +%T
./venv/bin/python3 -u -m rtv_solver.pipeline.rho_val_horizon_runs --batch_interval 200 --step_size 100 --cardinality 2 --seed 4242 --keep_active false --runs 1
date +%T
