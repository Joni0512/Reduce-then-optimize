#!/bin/bash
# Clean SIL checkpoint for SRL/twin-critic actor init - see chat: the previous checkpoint
# (sil_training_bi200_ss100_mixed_balanced_legacy_mlp_seed1) used an override train/val split
# that OVERLAPPED with srl_train_val_test_split.py's VAL_INSTANCES/TEST_INSTANCES (4 of 9 VAL
# instances were in its SIL-train set, 4 of 9 TEST instances were in its SIL-val set) - data
# leakage into every twin-critic run that used it as actor_checkpoint. This version trains only
# on 32 of the 38 TRAIN_INSTANCES, validating on the other 6 (also from TRAIN_INSTANCES) - no
# instance here ever touches VAL_INSTANCES or TEST_INSTANCES.
set -eu

echo "=== [$(date +%H:%M:%S)] SIL bi200/ss100 clean SRL-split (no VAL/TEST leakage) legacy MLP seed 1 ==="
./venv/bin/python3 rtv_solver/main.py \
  --mode coaml \
  --input_dir "solutions/li_lim/manifests/" \
  --override_training_files "lc101,lc102,lc105,lc106,lc107,lc109,lc201,lc203,lc206,lc207,lr101,lr102,lr105,lr106,lr108,lr109,lr110,lr112,lr201,lr203,lr206,lr207,lr209,lr210,lrc101,lrc103,lrc106,lrc107,lrc201,lrc203,lrc206,lrc207" \
  --override_validation_files "lc104,lc204,lr104,lr204,lrc104,lrc204" \
  --batch_interval 200 --step_size 100 --max_cardinality 2 \
  --learning_rate 0.0001 --epochs 5 \
  --imitation_scoring_rule legacy \
  --use_request_pruner False --use_request_graph_pruner False \
  --model_type mlp \
  --seed 1 \
  --output_dir "outputs/sil_training_bi200_ss100_clean_srl_split_legacy_mlp_seed1" \
  > "sil_training_bi200_ss100_clean_srl_split_legacy_mlp_seed1.log" 2>&1
echo "=== [$(date +%H:%M:%S)] DONE: clean SRL-split legacy MLP seed 1 (exit $?) ==="
