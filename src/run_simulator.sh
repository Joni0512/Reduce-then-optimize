#!/bin/sh

RH=1 #rollong horizen
INTERVAL=300
mkdir -p "../output_format/$RH/"
nohup python main.py --server_url "http://127.0.0.1:5000/" --input_file "../../../format_simulator/payload.pkl" --out_put_dir "../output_format/$RH/" --interval $INTERVAL --rh_factor $RH --max_cardinality 4  &
echo $! > ../proc/osrm-$RH
