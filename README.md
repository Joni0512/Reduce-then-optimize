# Online-RTV

## Code example

```
from online_rtv_solver import OnlineRTVSolver

solver = OnlineRTVSolver()
result = solver.solve_rtv(payload)
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
    'time_matrix': "nxn array", 
    'manifests': []
    
}
```


### Running with a single new requests

```
{
    
    'pickup': {
        'booking_id': int,
        'action': 'pickup',
        'am': int,
        'wc': int,
        'time_window_start': int,
        'time_window_end': int,
        'node_id': int,
        'loc': {'lat': float, 'lon': float, 'node_id': int}
    }, 
    'dropoff': {
        'booking_id': int,
        'action': 'dropoff',
        'am': int,
        'wc': int,
        'time_window_start': int,
        'time_window_end': int,
        'node_id': int,
        'loc': {'lat': float, 'lon': float, 'node_id': int}
    }, 
    'booking_id': "booking id of new requests", 
    
}
```

### Running with a batch of requests

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


# rolling-horizen-RTV

## Running

- Set up an osrm server. Follow https://github.com/Project-OSRM/osrm-backend
- `cd` into the src folder.
- run `python main.py --server_url "" --input_file "" --out_put_dir "" --interval 300 --rh_factor 0 --max_cardinality 4`
- Required parameters:
    - server_url: Url of the OSRM server (ex: "http://127.0.0.1:5000/")
    - input_file: path to the payload.pkl file
    - out_put_dir: directory to record outputs
    - interval: interval for the rolling horizon and batching
    - rh_factor: rolling horizon factor
    - max_cardinality: meximum size of shared trips
