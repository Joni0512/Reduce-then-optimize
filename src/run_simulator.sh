#!/bin/sh

RH=1   #rollong horizen
INTERVAL=600
OUTPUT_DIR="../output_wilson_orser/$RH/$INTERVAL"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/manifests"
nohup python main.py --server_url "http://127.0.0.1:50000/" --input_file "../inputs/wilson_nc_new.pkl" --out_put_dir "$OUTPUT_DIR/" --interval $INTERVAL --rh_factor $RH --max_cardinality 4  > "$OUTPUT_DIR/out.log" 2>&1 &

echo $! > "$OUTPUT_DIR/pid"
