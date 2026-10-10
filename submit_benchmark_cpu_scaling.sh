#!/bin/bash
# 2026-10-10: submits the CPU-scaling benchmark with 1, 2, 4, 8, 16 and 32 CPUs. Series 1: pool size 16 everywhere (what real runs use, so
# fewer CPUs = oversubscription); series 2: pool size = number of CPUs. 12 short jobs (each a few minutes, 1-32 CPUs, 40 min limit).
set -eu
cd "$(dirname "$0")"
for k in 1 2 4 8 16 32; do
  sbatch --cpus-per-task=$k --job-name=bench_c${k}_t16 --export=ALL,THREADS=16 submit_benchmark_cpu_scaling_serial.sbatch
  [ "$k" != "16" ] && sbatch --cpus-per-task=$k --job-name=bench_c${k}_tK --export=ALL,THREADS=$k submit_benchmark_cpu_scaling_serial.sbatch
done
