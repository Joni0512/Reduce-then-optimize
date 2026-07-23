#!/bin/bash
# 2026-07-17: RHO-no-learning baseline logs for bi200/ss40 (ratio 5:1, same
# ratio as bi400/ss80 but smaller horizon - completes the 2x2 horizon x
# ratio grid: {200,400} x {2:1, 5:1}).
#
# 2026-07-18: made resume-capable + hard-timeout-capped after LR2-class
# instances (lr206, then lr208) hung far past the RTV_TIMEOUT=1800s
# config-level backstop - that check only fires between certain RTV
# generation phases, so a single oversized phase can block past it
# indefinitely. A manual bash timeout (background sleep+kill, see run_with_timeout
# below) hard-kills after 10 minutes instead, so no single instance can
# block the batch indefinitely. NOTE: the GNU `timeout` command does NOT
# exist on macOS by default (no coreutils) - a first version of this script
# used it directly and every call failed instantly with exit 127 ("command
# not found"), silently marking 21 instances as "timed out" when they were
# never actually run at all. Fixed by implementing the timeout in pure bash.
# Also: previously, an instance killed mid-run could still get its
# PARTIAL assignment_data.jsonl copied into the final output (found on
# lr206: 14 lines instead of the ~25 a complete run produces) because the
# copy step only checked "does a run_dir exist", not "did the run finish
# cleanly". Now only copies on a clean (exit 0) run.

# Portable timeout: runs "$@" in the background, kills it (and its process
# group) if it's still alive after TIMEOUT_SECONDS. Returns the wrapped
# command's exit code, or 124 (matching GNU timeout's convention) on kill.
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

BI=200
SS=40
STAGING_OUT="staging_bi${BI}_ss${SS}"
FINAL_BASE="outputs/eval_rh_no_learning_bi${BI}_ss${SS}"
TIMEOUT_SECONDS=600

total=$(echo $INSTANCES | wc -w)
count=0
skipped=0
timed_out=()

for inst in $INSTANCES; do
  count=$((count + 1))

  # Resume: skip instances that already have a complete log from a prior run.
  if [ -f "${FINAL_BASE}/${inst}/rh_no_learning/assignment_data.jsonl" ]; then
    skipped=$((skipped + 1))
    echo ">>> [$count/$total] bi=${BI} ss=${SS} / $inst (skip - already done)"
    continue
  fi

  echo ">>> [$count/$total] bi=${BI} ss=${SS} / $inst"

  run_with_timeout ./venv/bin/python3 rtv_solver/main.py \
    --mode offline \
    --input_file "solutions/li_lim/manifests/${inst}.json" \
    --input_dir "solutions/li_lim/manifests/" \
    --imitation_solution_file "solutions/li_lim/manifests/${inst}.json" \
    --use_request_graph_pruner False \
    --use_request_pruner False \
    --seed 42 \
    --max_cardinality 2 --batch_interval "$BI" --step_size "$SS" \
    --output_dir "${STAGING_OUT}/${inst}" \
    > /dev/null 2>&1
  exit_code=$?

  if [ "$exit_code" -ne 0 ]; then
    echo "!!! $inst timed out or failed (exit $exit_code) after ${TIMEOUT_SECONDS}s - skipping, no partial data copied"
    timed_out+=("$inst")
    rm -rf "${STAGING_OUT}/${inst}"
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

echo "=== DONE: ${count}/${total} instances processed, ${skipped} skipped (already done) ==="
if [ ${#timed_out[@]} -gt 0 ]; then
  echo "=== ${#timed_out[@]} instance(s) timed out and were excluded: ${timed_out[*]} ==="
fi
