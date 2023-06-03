#!/bin/sh

NOVEH=20 # Number of vehicles
CAP=4 # Capacity of vehicles
RH=0 #rollong horizen
INTERVAL=300
mkdir -p "../output_osrm/$NOVEH/$CAP/$RH/"
nohup python main.py --server_url "http://127.0.0.1:5000/" --vehicle_file "../inputs/vehicles.csv" --request_file "../inputs/requests.csv" --out_put_dir "../output_osrm/$NOVEH/$CAP/$RH/" --interval $INTERVAL --rh_factor $RH --max_number_of_vehicles $NOVEH --max_capacity $CAP --max_cardinality 4  &
echo $! > osrm-$NOVEH-$CAP-$RH
