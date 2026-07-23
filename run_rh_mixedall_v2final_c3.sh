#!/bin/bash
# RHO + GNN pruner (mixed_all, v2_final), cardinality 3, threshold 0.5.
set -u

INSTANCES="lc108 lc109 lc207 lc208 lr111 lr112 lr210 lr211 lrc107 lrc108 lrc207 lrc208"
V2_MODEL="outputs/models_v2_gnnv2/rgnn_mixed_all_pw3_v2_final/rgnn_mixed_all_pw3_v2_final_best_val_f3.pt"

for inst in $INSTANCES; do
  echo ">>> rh/mixedall_v2final_c3_t05/${inst}"
  time python rtv_solver/main.py \
    --mode offline \
    --input_file "solutions/li_lim/manifests/${inst}.json" \
    --input_dir "solutions/li_lim/manifests/" \
    --imitation_solution_file "solutions/li_lim/manifests/${inst}.json" \
    --use_request_graph_pruner True \
    --request_graph_model_path "$V2_MODEL" \
    --request_graph_threshold 0.5 \
    --max_cardinality 3 --batch_interval 400 --step_size 100 \
    --output_dir "new_tests/cardinality3_compare/bi400_ss100/rh_mixedall_v2final_t05/${inst}"
done

echo "=== RH MIXED_ALL V2FINAL C3 T05 DONE (12/12) ==="
