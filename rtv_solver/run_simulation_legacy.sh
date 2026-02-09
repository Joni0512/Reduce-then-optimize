#!/bin/sh
# shell script to run main.py (possibly does not work atm, but main.py has feature parity)
RH=0 #rolling horizen
INTERVAL=600
echo "Debug Run"
mkdir -p "../output_format/$RH/$INTERVAL/"
nohup python main_legacy.py --server_url "http://127.0.0.1:5001/" --input_file "../inputs/localDB_payload_oct.pkl" --out_put_dir "../output_format/$RH/$INTERVAL/" --interval $INTERVAL --rh_factor $RH --max_cardinality 4  > ../output_format/debug.out 2> ../output_format/debug.err &
echo $! > ../output_format/processID-$RH-$INTERVAL