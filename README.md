# Online-RTV

# Installation

```
pip install rtv-solver
```

## Code example

### Initialize

```
from rtv_solver import OnlineRTVSolver

# Initialize the RTV solver with the URL of the OSRM server
online_rtv_solver = OnlineRTVSolver("http://127.0.0.1:50000/")
```

### Check feasibility of time slots

```
payload = {
    "requests": [
    {
        'am': int,
        'wc': int,
        'time_windows' : [
            {'pickup_time_window_start': int, 'pickup_time_window_end': int, 'dropoff_time_window_start': int, 'dropoff_time_window_end': int,},
        ],
        'pickup_pt': {'lat': float, 'lon': float, 'node_id': int},
        'booking_id': int,
        'dropoff_time_window_start': int,
        'dropoff_time_window_end': int,
        'dropoff_pt': {'lat': float, 'lon': float, 'node_id': int}
    }],
    "driver_runs": driver_runs
}

feasibility = online_rtv_solver.check_feasibility(payload)


feasibility <-- [(feasible_window,vmt/pmt ratio)]
```

### Generating a manifest

```
current_time = 5*3600+30*60 # 05:30:00 pm
driver_runs, unserved_requests = online_rtv_solver.solve_rtv(current_time, new_payload)

unserved_requests <-- [list of ids of the requests that are not feasible to serve]
```

#### Fast option

```
driver_runs, unserved_requests = online_rtv_solver.solve_rtv_fast(new_payload)

unserved_requests <-- [list of ids of the requests that are not feasible to serve]
```

### Simulate the vehicles
```
current_time = 5*3600+40*60+00 # Simulate to 05:40:00 pm
new_driver_runs = online_rtv_solver.simulate_manifest(current_time, driver_runs)
```

## Payload format

### Common format
```
{
    
    'depot': {
        'loc': {'lat': float, 'lon': float, 'node_id': int}
    }, 
    'date': 'yyyy-mm-dd', 
    'driver_runs': [],
    'requests': []
    
}
```

### Requests

```
{
    
    'requests': [ {
        'am': int,
        'wc': int,
        'pickup_time_window_start': int,
        'pickup_time_window_end': int,
        'pickup_pt': {'lat': float, 'lon': float, 'node_id': int},
        'booking_id': int,
        'dropoff_time_window_start': int,
        'dropoff_time_window_end': int,
        'dropoff_pt': {'lat': float, 'lon': float, 'node_id': int}
    }] 
    
}
```

### DriverRun

```
{
    
    'DriverRun': [ {
        'state': {
            'run_id': int,
            'start_time': int,
            'end_time': int,
            'am_capacity': int,
            'wc_capacity': int,
            'locations_already_serviced': int,
            'locations_dt_seconds': int,
            'loc': {'lat': float, 'lon': float, 'node_id': int},
            'total_locations': int,
        },
        'manifest': Stop[list]
    }] 
    
}
```


### Stop

```
{
    
    'Stop': [ {
        'run': int,
        'booking_id': int,
        'order': int,
        'action': string,
        'loc': {'lat': float, 'lon': float, 'node_id': int}
        'scheduled_time': int,
        'am': int,
        'wc': int,
        'time_window_start': int,
        'time_window_end': int,
    }] 
    
}
```

# Set up the OSRM Server

```
wget https://download.geofabrik.de/north-america/us/north-carolina-latest.osm.pbf
docker run -t -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend osrm-extract -p /opt/car.lua /data/north-carolina-latest.osm.pbf || echo "osrm-extract failed"
docker run -t -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend osrm-partition /data/north-carolina-latest.osrm || echo "osrm-partition failed"
docker run -t -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend osrm-customize /data/north-carolina-latest.osrm || echo "osrm-customize failed"
docker run -t -i -p 5000:5000 -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend osrm-routed --algorithm mld /data/north-carolina-latest.osrm
```

# Building

```
python -m build
twine upload dist/rtv_solver-[version]*

```
