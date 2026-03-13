
[System Overview](#system-overview-and-main-components)

[Run](#code-example)


# RTV solver - System Overview

The codebased implements both an online and an offline request-trip-vehicle (RTV) solver with an Integer Linear Programme (ILP) solving a Pickup and Delivery Problem with Time Windows (PDPTW). All core software components are organised in the `rtv_solver` package. 
The codebase is used for research purposes and experimental runs with different data and config settings.
The codebase is used to implement a new approach integrating machine learning (ML) or specifically structured reinforcement learning (SRL) to improve the assignment of the RTV approach in long-term horizons.

Execution entry points:
- Code execution via `rtv_solver/main.py`. 
- CLI / Batch execution via shell script `rtv_solver/run_main.sh`

The implementation covers the full solver pipeline, including payload parsing, trip generation, feasibility checks, optimization via Gurobi, and result output.

The script `rtv_solver/main.py` handles the argument parsing to guarantee reproducible experiments, collects the input data and instantiates logging.

The script either calls the online solver in `rtv_solver/online_rtv_solver.py` that solves the entire payload that was given to it. 
As an alternative, the script calls the offline solver in `rtv_solver/online_rtv_solver.py` that generates a solution by splitting up payload batches and incrementing the simulation time of the vehicles independently. 

## Main components
- The folder `rtv_solver/handlers/` implements the core solver logic parsing data, handling vehicles and requests, and generating trips and assigning the trips for the optimal assignment.
  - `payload_parser.py`
  - `request_handler.py`
  - `vehicle_handler.py`
  - `trip_handler.py`
  - `network_handler.py`
  - `stats_parser.py`
- The folder `rtv_solver/structure` implements the data classes and data models for low-level functionality.
  - `Trip`, `SharedTrip`, `TripCost`
  - `Vehicle`, `Request`, `Node`, 
  - `Sequence`, `VehicleStop`
- The folder `rtv_solver/tests` stores the tests that are based on `pytest`. The focus is on general integration tests to continually check overall functional parity and backward compatibility.
- The folder `rtv_solver/util` implements basic logging and some helper functions.
- The folder `rtv_solver/visuals` implements basic visualisations to understand solver behavior and analyse possible improvements.

## High-level flow

1. Payload data is loaded using pickle files in a JSON format.
2. The `payload_parser.py` parses the data for usage in the solver.
3. From the payload, the `request_handdler.py` collects all requests and generates data objects of these requests. A request is defined by a pickup and a dropoff and the according time windows restricting the feasibility of trips.
4. The `vehicle_handler.py` collects all vehicles and generates data objects of these vehicles. A vehicle is defined by its operating time, its next position, the time of its next position and possibly a sequence of stops to finish the assigned trips. 
5. The `trip_handler.py` generates on-demand trips from each request. For shared trips, it checks the feasibility of different on-demand trips in sequence. A trip is defined by a sequence of `vehicle_stops`. 
6. For both on-demand trips and shared trips, the `vehicle_handler.py` generates trip costs per vehicle based on the distance of a route and checks the feasibility with already existing obligation of boarded requests.
7. The routes are created using the `network_handler.py` that build API requests to a separate open source routing machine (OSRM) backend server. It also parses the responses to be used subsequently.
8. The `trip_handler.py` handles the final assignment using a Gurobi ILP solver minimizing the costs while ensuring that each request is only assigned once and each vehicle only handles one trip.
9. The main script handles the manifests of stops, simulation steps and the output for logging.


# Code example

## Run solver

[Shell script run_main.sh](rtv_solver/run_main.sh) abstracts all details to run the basic script automatically.
```sh
#!/usr/bin/env bash
set -e
INPUT_FILE="wilson_nc_initial.pkl" # stored in rtv_solver/inputs/
python main.py \
  --server_url "http://127.0.0.1:5001/" \
  --input_file $INPUT_FILE \
  --max_cardinality 4 \
  --batch_interval 3600 \
  --step_size 1200
echo "Run complete"
```

In order to edit or customize changes, the main script [`rtv_solver/main.py`](rtv_solver/main.py) is the main entry point where the basic changes can be viewed. In order to run the code based on a prior config, you can run [Shell script reproduce_run.sh](rtv_solver/reproduce_run.sh).

## Visualise results

In order to show all routes of the trip, you can run the `rtv_solver/visuals/route_manifest_mapper.py` with an updated folder to see the trips and routes of all vehicles. This file still needs updates to be useable for actual anaylsis purposes.

```py
from pathlib import Path
from rtv_solver.visuals.route_manifest_mapper import RouteManifestMappper
from rtv_solver.structure.config import Config

folder = Path("../outputs/debug/run_20260208_142128_5706bf")     
with open(folder / "result_driver_runs.json", 'r') as driver_runs_file:
    loaded_data = json.load(driver_runs_file)
config_file = load_json(folder / "config.json")
config = Config.from_dict(config_file["config_dict"])

mapper = RouteManifestMapper(config)
geojson = mapper.manifest_to_geojson(loaded_data, 18)
mapper.save_geojson(geojson, folder / "route_manifest.geojson")
```


# Payload format

There exist two separate payload format that work with the existing codebase on branch `rh-ml` 
- `wilson`
- `chattanooga`

Each format can be used as there is an automatic detection and conversion of the payloads when using the `main.py` script.

### Wilson

The JSON format is mainly for documentation purposes and facilitates understanding instead of usability. For `wilson`, the following format applies (example `inputs/wilson_nc_initial.pkl`):
The value `node_id` might not be part of the original data object, but is generally added wherever locations are handled.

``` js 
{
   "requests": [ list of requests
        {
            "booking_id": "1", 
            "pickup_pt":{
                "lon":-77.930793762,
                "lat":35.780387878,
                "node_id": 12
            },
            "dropoff_pt":{
                "lon":-77.893867493,
                "lat":35.719944,
                "node_id": 13
            },
            "pickup_time_window_start":20043,
            "pickup_time_window_end":21843,
            "dropoff_time_window_start":20654.9,
            "dropoff_time_window_end":22454.9,
            "am":1,
            "wc":0
        }
   ],
   "depot": {
        "pt":{
            "lat":35.723017652422435,
            "lon":-77.90871990823223
        },
        "node_id":0
    },
    "driver_runs": [ list of vehicles
        {
        "state": { initial state
            "run_id":0,
            "start_time":18000,
            "end_time":72000,
            "am_capacity":8,
            "wc_capacity":3,
            "locations_already_serviced":0,
            "location_dt_seconds":0,
            "total_locations": 4,
            "loc":{
                "lat":35.723017652422435,
                "lon":-77.90871990823223,
                "node-id": 0
            }
        },
        "manifest":[ list of all actual stops of that vehicle
            "run_id": 0 (corresponding vehicleID),
            "order": 5 (order in trip),
            "action": ["pickup", "dropoff", "depot"] (options),
            "booking_id": 1 (corresponding requestID),
            "loc": {
                "lat":35.723017652422435,
                "lon":-77.90871990823223,
                'node_id': 1
            }
            "am":1,
            "wc":0,
            "scheduled_time": 20567 (actual arrival time at the stop)
            "time_window_start": 20000 (range of arrival_time, copied from time windows),
            "time_window_end": 21000 (range of arrival_time)
        ]
        }
    ]
}
```

### Chattanooga

The structure used in Chattanooga is different to the one used in `wilson`'.
- Smaller set of key-value pairs in `driver_runs`, no separation of `state` and `manifest`
- Addition of `date`
- Addition of `time_matrix` with arrays of time values (pre-calculation of the distances between all possible nodes)
- Addition of `requests - "pickup_node_id" ^ "dropoff_node_id"` (nodes are IDs for pre-calculated distances and match array index)

``` js
{
    "driver_runs":[
        {
            "run_id":0,
            "start_time":14400,
            "end_time":54000,
            "am_capacity":8,
            "wc_capacity":3
        }
    ],
   "requests":[
        {
            "am":0,
            "wc":1,
            "pickup_time_window_start":14400,
            "pickup_time_window_end":16200,
            "dropoff_time_window_start":15300,
            "dropoff_time_window_end":18600,
            "pickup_pt":{
                "lat":35.045644,
                "lon":-85.319982
            },
            "dropoff_pt":{
                "lat":35.022033,
                "lon":-85.241765
            },
            "booking_id":1129707,
            "pickup_node_id":1,
            "dropoff_node_id":2
        }
   ],
   "depot":{
        "pt":{
            "lat":35.723017652422435,
            "lon":-77.90871990823223},
        "node_id":0
    },
    "date": "2023-10-19",
    "time_matrix": [
        [
            0,
            575,
            ...
        ],
        [
            527,
            0,
            ...
        ]
    ]
}
```

# Development / Build / Run Notes

## Installation

```
pip install rtv-solver
```
Gurobi-License is required for local runtime.

If one wants to debug the codebase or develop new features, one needs to make sure that the user runs the python package not from the lastest version of the online package, but rather from their local version. Caution on the difference between rtv-solver as the online package and rtv_solver as the packaging in the repository. If it has already been installed, update with the following commands for editable runs that incorporate one's changes.
```
pip uninstall rtv-solver
pip install -e . (from rtv-solver directory)
```

## Calculation of optimal results

For results based on LiLimParser in the right data format in order for us to use it, we transformed the data.
The data is too big for Github but can easily be recreated by calling the '__main__' in [`rtv_solver/parser/li_lim_parser.py`](rtv_solver/parser/li_lim_parser.py)

## Set up OSRM backend server

Depending on the input data, you are using and the main operating area, you have to adapt the basis of the OpenStreetMap routing data for the backend server.

`wilson` is located in North Carolina, USA (center: 35°43′53″N 77°55′43″W)

`chattanooga` is located in Tennessee, USA (center: 35.065958°N 85.248386°W)

### Preprocessing
```bash
wget https://download.geofabrik.de/north-america/us/north-carolina-latest.osm.pbf
docker run -t -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend osrm-extract -p /opt/car.lua /data/north-carolina-latest.osm.pbf || echo "osrm-extract failed"
docker run -t -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend osrm-partition /data/north-carolina-latest.osrm || echo "osrm-partition failed"
docker run -t -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend osrm-customize /data/north-carolina-latest.osrm || echo "osrm-customize failed"
docker run -t -i -p 5000:5000 -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend osrm-routed --algorithm mld /data/north-carolina-latest.osrm
```

### Starting up server

The last command must be called from the directory where the unpacked data is located. This initializes the Docker container and starts it.
```bash
docker run -d -p 5001:5000 \
  -v "${PWD}:/data" \
  ghcr.io/project-osrm/osrm-backend \
  osrm-routed --algorithm mld /data/north-carolina-latest.osrm
```
In order to test the working OSRM backend server, you can put the following information into your lcoal browser and it should work immediately giving you some value for the routing request

[Click for local test](http://localhost:5001/route/v1/driving/-77.90871990823223,35.723017652422435;-77.90,35.75)


## Testing
In order to check that the basic functionality of the different solvers works including certain config combinations, one must install `pytest` and it can run all available tests from `../tests` automatically with the following command in your folder structure. It is important that each method and each files are named with `test_` as a prefix to the specific functions. The project uses `pytest.mark` to filter tests. The marks can be viewed in the `pyproject.toml`.

```py
pip install pytest
pytest -q
# alternative with marks in order to filter only certain tests (e.g. integration)
pytest -q -m integration
```

## Runtime on CPU cluster

The following commands help to set up the codebase on a CPU cluster.
1. Get access to cluster via Ticketing system (approval by supervisor), e.g. [Cornell Unicorn cluster](https://it.coecis.cornell.edu/researchit/using-the-unicorn-cluster/)
2. Install gurobi and hexaly optimizers with the respective licenses.
6. initialize conda for other package management with `conda env create -f rtv-solver/environment.yml -n coaml`
10. Install rtv-solver package as editable from `rtv-solver` with `pip install -e .`
11. If `gurobipy` or `hexalypy` fails, install them again through pip with the right version from the 
12. The files that you use for the runtime on the server need to have the time matrix as part of the payload file. This needs to be prepared offline during the creation of the files. For most clusters, it is not allowed to have a OSRM backend server running. This is explained in a different section <ADD_LINK_HERE>. You can make sure that it will work by turning off the backend server and running a payload.

Attention: it is not possible to run the entire software based on the OSRM backend server, thus the local files need to be appended with the travel_time_matrix in order to run properly.

### Installation of Gurobi

Besides local Gurobi license for your local machine, one needs a Academic WLS License. 
1. wget https://packages.gurobi.com/13.0/gurobi13.0.1_linux64.tar.gz 
4. (cd to gurobi folder) unpack installation file `tar -xvzf 'gurobi-file'`
5. (cd into gurobi > bin) `./grgetkey 'license file'`
set PATH variables for Gurobi installation
8. run script again with `source ~/.bashrc` to update shell environment
9. Command `gurobi_cl --license` should confirm the successful installation of Gurobi Optimizer WSL.

### Installation of Hexaly

Request Server License from hexaly
1. `wget https://www.hexaly.com/downloads/14_5_20260220/Hexaly_14_5_20260220_Linux64.run`
2. `bash Hexaly_14_5_20260220_Linux64.run --nointeractive --noroot`
3. `pip install hexaly -i https://pip.hexaly.com`
4. Download the license.dat file from Hexaly website and add it to the hexaly installation. (hexaly_14_5 folder)
5. Open PATH editor `nano ~/.bashrc`
6. Add the following line to the end of the file. `export HX_LICENSE_PATH="$HOME/hexaly_14_5/license.dat"`
7. Update PATH variables with `source ~/.bashrc`



# Future Development

# Behavior Understanding

Offline solver: Currently, the optimization is built in order to create all possible RTV combinations (up to max_cardinality) that exist in the payload, this optimization is run once. This leads to the weird behavior that a longer batch_size with more requests maxxes out the capacity of the vehicle up to max_cardinality but smaller increments of batch_size are able to serve more requests as they run the same optimization twice during that time. This is rather a characteristic of the approach in general

## Issues

No major problems with the RTV solver. 

## Todos & Improvements

- [ ] repetitions in structures and data objects, DriverRun, Vehicle and driver_run-dicts - major effort but probably very valuable to align the two and only switch to dictionaries at specific place (could later improve conditionals and when data can be changed to notice problems earlier and could remove certain issues)
- [ ] test (& fix) rebalancing (not as important in current setup)
- [ ] Dropoff window in `wilson / rh-ml` is fixed in payload, add a method to define this in main_script and update the payload data during the call based on our own definition of waiting times that we want to consider. (alternative: create new payloads with these changes)
- [ ] add feature to tag vehicles as "inactive" in contrast to "started" and only calculate trip Generation with active vehicles when they are not used anymore
- [ ] ensure that one can still build a python package from it without setup.py and use pyproject.toml more usefully
- [ ] export a requirements.txt and integrate to pyproject.toml
- [ ] when breaking RTV generation time, break off new generation but still optimize to keep it running but with a warning in the stats that it did not run to optimality (instead of crashing on the spot - it would be interesting to see the same run with all the trips already created)
- [ ] check TODO, FIXME, NOTE in the code
- [ ] VehicleAssignment is indeterministic (different vehicles handle same requests, same stats so no major issue)
