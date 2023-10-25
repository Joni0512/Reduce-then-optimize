#!/bin/sh

RH=0 #rollong horizen
INTERVAL=600
mkdir -p "../output_format/$RH/$INTERVAL/"
nohup python main.py --server_url "http://127.0.0.1:5000/" --input_file "../inputs/localDB_payload_oct.pkl" --out_put_dir "../output_format/$RH/$INTERVAL/" --interval $INTERVAL --rh_factor $RH --max_cardinality 4  &
echo $! > osrm-$RH-$INTERVAL
