### Initialize

```
from rtv_solver import OnlineRTVSolver

# Initialize the RTV solver with the URL of the OSRM server
online_rtv_solver = OnlineRTVSolver("http://127.0.0.1:50000/")
```

### Check feasibility of time slots

```py
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

```py
current_time = 5*3600+30*60 # 05:30:00 pm
driver_runs, unserved_requests = online_rtv_solver.solve_pdptw_rtv(new_payload)

unserved_requests <-- [list of ids of the requests that are not feasible to serve]
```

#### Fast option with Insertion Heuristic

```py
driver_runs, unserved_requests = online_rtv_solver.solve_pdptw_heuristic(new_payload)

unserved_requests <-- [list of ids of the requests that are not feasible to serve]
```

#### Serve a request as soon as possible

```py
new_payload = {
    "depot": {},
    "requests": [],
    "driver_runs": []
}

driver_runs = online_rtv_solver.serve_asap(new_payload)
```

### Simulate the vehicles
```py
current_time = 5*3600+40*60+00 # Simulate to 05:40:00 pm
new_driver_runs = online_rtv_solver.simulate_manifest(current_time, driver_runs)
```

### Regenerating a manifest

```py
payload = {
    "driver_runs": driver_runs,
    "depot": depot
}

driver_runs = online_rtv_solver.resolve_pdptw_rtv(payload)
```

### Get stats from the manifest

```js
depot = {
    "pt": {"lat": val, "lon": val}
}

driver_runs <- generated driver runs

feasibility, stats = online_rtv_solver.get_stats(depot, driver_runs)

feasibility {true or false} <- Is this schedule feasibile

stats = {
    "vmt": ,
    "pmt": ,
    "serviced": ,
    "wait_time": [wait time for each request],
    "detour": [detour for each request]
}
```

# Building

```
python -m build
twine upload dist/rtv_solver-[version]*
```