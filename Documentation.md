# Documentation

## JSON object structure

File: `inputs/wilson/random_weekeday_2.pkl`
Structure used in code on branch `wilson` and `rh-ml`

Manifest is used for all trips that have actually been carried out (probably in order to analyse the routes taken aftwards)

``` js
{
   "requests":[
        {
            "booking_id":"1",
            "pickup_pt":{
                "lon":-77.930793762,
                "lat":35.780387878
            },
            "dropoff_pt":{
                "lon":-77.893867493,
                "lat":35.719944
            },
            "pickup_time_window_start":20043,
            "pickup_time_window_end":21843,
            "dropoff_time_window_start":20654.9,
            "dropoff_time_window_end":22454.9,
            "am":1,
            "wc":0
        }
   ],
   "depot":{
        "pt":{
            "lat":35.723017652422435,
            "lon":-77.90871990823223
        },
        "node_id":0
    },
    "driver_runs":[
        {
        "state":{
            "run_id":0,
            "start_time":18000,
            "end_time":72000,
            "am_capacity":8,
            "wc_capacity":3,
            "locations_already_serviced":0,
            "location_dt_seconds":0,
            "loc":{
                "lat":35.723017652422435,
                "lon":-77.90871990823223
            }
        },
        "manifest":[]
        }
    ]
```


File `inputs/localDB_payload_oct.pkl`

The structure used in Chattanooga is different to the one used in `wilson`'.
- Smaller set of `driver_runs` key-value pairs, no separation of `state` and `manifest`
- Addition of `date`
- Addition of `time_matrix` with arrays of time values (assumption: pre-calculation of the distances between all possible nodes)
- Addition of `requests - "pickup_node_id" ^ "dropoff_node_id"` (assumption: nodes are pre-calculated distances )
- (Order is different to the JSON, but does not bother its processing.)

Assumptions validated!

`time_matrix`: asymmetric 385 x 385 matrix, main diagonal is always 0 and transposed values are similar (meaning: A-B not equal distance to B-A)
192 requests with a 2 nodes (pickup / dropoff) and 1 depot - 192 * 2 + 1 = 385 combinations
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
            "lon":-77.90871990823223
        },
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
```