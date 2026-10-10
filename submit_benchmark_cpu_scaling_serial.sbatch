#!/bin/bash
#SBATCH --job-name=bench_cpu
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --qos=cm4_serial
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=bench_cpu_%j.log
#SBATCH --error=bench_cpu_%j.err

# 2026-10-10: CPU-scaling benchmark (see scripts/benchmark_cpu_scaling.py). CPUs are overridden at submit time with
# --cpus-per-task=K (the header value is only a default); THREADS (exported by submit_benchmark_cpu_scaling.sh) sets Config.MAX_THREAD_CNT.
# Same work in every job; compare the "TOTAL" line of the logs.

cd "$SLURM_SUBMIT_DIR" || exit 1
export GRB_LICENSE_FILE="$HOME/.gurobi/gurobi.lic"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK THREADS=${THREADS:-16} node=$(hostname)"
./venv/bin/python3 -u scripts/benchmark_cpu_scaling.py --threads "${THREADS:-16}" --windows 5 lr107 lrc205 lc202 lr202
