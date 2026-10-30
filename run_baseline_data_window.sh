#!/bin/bash
# 2026-10-29: generic RHO-no-learning baseline (label source for the request-pruner dataset) for ANY window: ./run_baseline_data_window.sh <batch_interval> <step_size> [shard] [nshards]
# Copy of run_baseline_data_bi200_ss40.sh (resume + hard timeout, copy only on a clean exit) with these changes: (1) BI/SS from the arguments, (2) hard timeout 1800 s per instance
# (CLAUDE.md rule: single solver instance max. 30 min; the original used 600 s), (3) --max_thread_cnt 4 (main.py default is 16, the cluster jobs get 4 CPUs; no effect on results), (4) optional sharding (instance i is run by shard i % nshards) so one window can run as several parallel cluster jobs.
# Output: outputs/eval_rh_no_learning_bi<BI>_ss<SS>/<inst>/rh_no_learning/{assignment_data.jsonl,config.json}; cardinality 2, seed 42, no pruners (same flags as the 200/40 baseline).
run_with_timeout() {
  "$@" &
  local cmd_pid=$!
  (
    sleep "$TIMEOUT_SECONDS"
    if kill -0 "$cmd_pid" 2>/dev/null; then
      # kill multiprocessing worker children first (they're children of
      # cmd_pid, not grandchildren - pkill -P catches them directly),
      # then the main process itself.
      pkill -9 -P "$cmd_pid" 2>/dev/null
      kill -9 "$cmd_pid" 2>/dev/null
    fi
  ) &
  local watcher_pid=$!
  wait "$cmd_pid" 2>/dev/null
  local exit_code=$?
  kill "$watcher_pid" 2>/dev/null
  wait "$watcher_pid" 2>/dev/null
  return $exit_code
}
set -u

INSTANCES="lc101 lc102 lc103 lc104 lc105 lc106 lc107 lc108 lc109 lc201 lc202 lc203 lc204 lc205 lc206 lc207 lc208 lr101 lr102 lr103 lr104 lr105 lr106 lr107 lr108 lr109 lr110 lr111 lr112 lr201 lr202 lr203 lr204 lr205 lr206 lr207 lr208 lr209 lr210 lr211 lrc101 lrc102 lrc103 lrc104 lrc105 lrc106 lrc107 lrc108 lrc201 lrc202 lrc203 lrc204 lrc205 lrc206 lrc207 lrc208"

BI=$1
SS=$2
SHARD=${3:-0}
NSHARDS=${4:-1}
STAGING_OUT="staging_bi${BI}_ss${SS}_shard${SHARD}"
FINAL_BASE="outputs/eval_rh_no_learning_bi${BI}_ss${SS}"
# 2026-10-30: timeout and instance list overridable (user asked to raise the 30 min cap so the label data become complete: the first pass left 9/4/3/1 instances out at 40/10, 100/25, 120/30, 200/50)
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1800}
INSTANCES=${INSTANCES_OVERRIDE:-$INSTANCES}

total=$(echo $INSTANCES | wc -w)
count=0
skipped=0
timed_out=()

for inst in $INSTANCES; do
  count=$((count + 1))
  if [ $(( (count - 1) % NSHARDS )) -ne "$SHARD" ]; then
    continue
  fi

  # Resume: skip instances that already have a complete log from a prior run.
  if [ -f "${FINAL_BASE}/${inst}/rh_no_learning/assignment_data.jsonl" ]; then
    skipped=$((skipped + 1))
    echo ">>> [$count/$total] bi=${BI} ss=${SS} / $inst (skip - already done)"
    continue
  fi

  echo ">>> [$count/$total] bi=${BI} ss=${SS} / $inst ($(date '+%H:%M:%S'))"

  run_with_timeout ./venv/bin/python3 rtv_solver/main.py \
    --mode offline \
    --input_file "solutions/li_lim/manifests/${inst}.json" \
    --input_dir "solutions/li_lim/manifests/" \
    --imitation_solution_file "solutions/li_lim/manifests/${inst}.json" \
    --use_request_graph_pruner False \
    --use_request_pruner False \
    --seed 42 --max_thread_cnt "${MAX_THREAD_CNT:-4}" \
    --max_cardinality 2 --batch_interval "$BI" --step_size "$SS" \
    --output_dir "${STAGING_OUT}/${inst}" \
    > /dev/null 2>&1
  exit_code=$?

  if [ "$exit_code" -ne 0 ]; then
    echo "!!! $inst timed out or failed (exit $exit_code) after at most ${TIMEOUT_SECONDS}s - skipping, no partial data copied"
    timed_out+=("$inst")
    rm -rf "outputs/${STAGING_OUT}/${inst}"
    continue
  fi

  run_dir=$(find "outputs/${STAGING_OUT}/${inst}" -maxdepth 2 -type d -name "${inst}_*" | head -1)
  if [ -z "$run_dir" ]; then
    echo "!!! could not find output dir for $inst despite clean exit, skipping copy"
    continue
  fi

  target_dir="${FINAL_BASE}/${inst}/rh_no_learning"
  mkdir -p "$target_dir"
  cp "${run_dir}/assignment_data.jsonl" "$target_dir/assignment_data.jsonl"
  cp "${run_dir}/config.json" "$target_dir/config.json"
  rm -rf "outputs/${STAGING_OUT}/${inst}"
done

echo "=== DONE bi=${BI} ss=${SS} shard ${SHARD}/${NSHARDS}: ${count}/${total} instances in list, ${skipped} skipped (already done) ==="
if [ ${#timed_out[@]} -gt 0 ]; then
  echo "=== ${#timed_out[@]} instance(s) timed out and were excluded: ${timed_out[*]} ==="
fi
