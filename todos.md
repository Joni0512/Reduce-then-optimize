# Major problems

## 1: Final boarded request is never dropped off
Possible Explanation: 
1. vehicle is deleted before final request is offboarded (fix not complete) - alternative: add a status that is always filtered but never delete vehicles
2. trip sequence is reset before it is fully carried out

### Print sequence to understand bug
``` js
Iteration  15 starting_time: 64800.0 latest_time: 67435 active_requests: 1 boarded_requests: 1
T: 67435 batch: 21
pickup:  14-535.0
dropoff:  13-522.0
debug
Iteration  16 starting_time: 67435 latest_time: 67435 active_requests: 1 boarded requests: 1
T: 71035.0 batch: 0
dropoff:  14-535.0
pickup:  15-577.0
debug
Iteration  17 starting_time: 71035.0 latest time: 67435 active_requests: 0 boarded_requests: 1
T: 74635.0 batch: 0
Vehicle 0 completed its run and is removed from the simulation.
debug
Iteration  18 starting_time: 74635.0 latest_time: 67435 active_requests: 0 boarded_requests: 1
T: 78235.0 batch: 0
Vehicle 0 completed its run and is removed from the simulation.
```

## 2: No shared requests / no multiple requests per time interval (deprio until normal code runs properly)
Context: Always only a single active request (process of being picked up) and at most one boarded request

Possible explanation:
1. Shared trips do not work
2. 

## 3: (probable) Solution is indeterministic but I cannot pinpoint it 
Context: Different requests seem to be picked up under the same arguments

## 4: Dropoff window in `wilson / rh-ml` fixed in payload
Context: Dropoff window is defined in the payload, should this not be overwritten in the code based on our own definition of waiting times that we want to consider. 

# Outstanding tests

## Rolling horizon

# Improvements
- check TODO, FIXME, NOTE in the code
- move all arguments required for a run into the .args file and store it with logs of a run (currently only debug mode is always fixed and it is not entirely cleared)
