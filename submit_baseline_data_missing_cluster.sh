#!/bin/bash
# 2026-10-30: second pass of stage 1: the instances that hit the 30 min cap in the first pass (user asked for complete label data), one job per (window, instance),
# 23 h hard timeout per instance inside a 24 h serial_std job (4 CPUs, 16 GB). Instances that still time out here need serial_long (7 days).
set -eu
cd "$(dirname "$0")"
sub() {  # $1=BI $2=SS $3..=instances
  bi=$1; ss=$2; shift 2
  for inst in "$@"; do
    sbatch --job-name=rh_base_${bi}_${ss}_$inst --time=24:00:00 --export=ALL,BI=$bi,SS=$ss,SHARD=0,NSHARDS=1,INSTANCES_OVERRIDE=$inst,TIMEOUT_SECONDS=82800 submit_baseline_data_window.sbatch
  done
}
sub 40 10 lc202 lc203 lc204 lr203 lr206 lr210 lrc204 lrc207 lrc208
sub 100 25 lc204 lr204 lr206 lr208
sub 120 30 lr204 lr206 lr208
sub 200 50 lr206
