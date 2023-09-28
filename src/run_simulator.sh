#!/bin/sh

RH=0 #rollong horizen
INTERVAL=600
mkdir -p "../output_format/$RH/$INTERVAL/"
nohup python main.py --server_url "http://127.0.0.1:5000/" --input_file "../../../format_simulator/payload.pkl" --out_put_dir "../output_format/$RH/$INTERVAL/" --interval $INTERVAL --rh_factor $RH --max_cardinality 4  &
echo $! > ../proc/osrm-$RH-$INTERVAL
