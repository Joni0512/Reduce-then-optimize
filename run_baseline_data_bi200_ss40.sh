#!/bin/bash
# 2026-07-17: RHO-no-learning baseline logs for bi200/ss40 (ratio 5:1, same
# ratio as bi400/ss80 but smaller horizon - completes the 2x2 horizon x
# ratio grid: {200,400} x {2:1, 5:1}).
set -u

INSTANCES="lc101 lc102 lc103 lc104 lc105 lc106 lc107 lc108 lc109 lc201 lc202 lc203 lc204 lc205 lc206 lc207 lc208 lr101 lr102 lr103 lr104 lr105 lr106 lr107 lr108 lr109 lr110 lr111 lr112 lr201 lr202 lr203 lr204 lr205 lr206 lr207 lr208 lr209 lr210 lr211 lrc101 lrc102 lrc103 lrc104 lrc105 lrc106 lrc107 lrc108 lrc201 lrc202 lrc203 lrc204 lrc205 lrc206 lrc207 lrc208"

BI=200
SS=40
STAGING_OUT="staging_bi${BI}_ss${SS}"
FINAL_BASE="outputs/eval_rh_no_learning_bi${BI}_ss${SS}"

total=$(echo $INSTANCES | wc -w)
count=0

for inst in $INSTANCES; do
  count=$((count + 1))
  echo ">>> [$count/$total] bi=${BI} ss=${SS} / $inst"

  ./venv/bin/python3 rtv_solver/main.py \
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

  run_dir=$(find "outputs/${STAGING_OUT}/${inst}" -maxdepth 2 -type d -name "${inst}_*" | head -1)
  if [ -z "$run_dir" ]; then
    echo "!!! could not find output dir for $inst, skipping copy"
    continue
  fi

  target_dir="${FINAL_BASE}/${inst}/rh_no_learning"
  mkdir -p "$target_dir"
  cp "${run_dir}/assignment_data.jsonl" "$target_dir/assignment_data.jsonl"
  cp "${run_dir}/config.json" "$target_dir/config.json"
done

rm -rf "outputs/${STAGING_OUT}"
echo "=== DONE: ${count}/${total} instances processed ==="
