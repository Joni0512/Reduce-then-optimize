# Major problems

## Rolling horizon (step_size < batch_interval) breaks after first iteration
Online and offline only work if step_size == batch_interval which is basically batched optimization but we do not have any overlapping state; normally the vehicle assignment should just be fixed and then we re-optimize based on new information (trips are reset entirely)
Explanations:
- what happens if a vehicle is assigned but outside of the step_size (normally this should just be regarded as a new inactive request)

## (probable) Solution is indeterministic but I cannot pinpoint it 
Context: Different requests seem to be picked up under the same arguments

## Dropoff window in `wilson / rh-ml` fixed in payload
Context: Dropoff window is defined in the payload, this should be overwritten in the code based on our own definition of waiting times that we want to consider. -> do a single run over all the requests to fix this problem once?

## Error: Sometimes requests are lost
Following settings lead to a breaking run in check_consistency_of_manifests
Reproducible settings: with wilson_data, cardinality = 3, thread_cnt = 16, batch_interval = 1800, step_size = 1800 (should be reproducible with this)

## Re-optimize if one vehicle has dropped off their steps (major effort)
Currently, the optimization is built in order to create all RTV combinations (up to max_cardinality) defined at the beginning and run optimize this once. This leads to the weird behavior that a longer batch_size with more requests maxes out the capacity of the vehicle up to max_cardinality but smaller increments of batch_size are able to serve more requests as they run the same optimization twice during that time.

# Improvements
- check TODO, FIXME, NOTE in the code
- move all arguments required for a run into the .args file and store it with logs of a run (currently only debug mode is always fixed and it is not entirely cleared)   
- time which effects cardinality and threads have on the performance of the code (should lead to a note which process we really need to improve with and do we rather want short batch_intervals or just short steps)
- add statistics and service rate, so we can compare the outcome of different runs
- output logging, so we have reproducible results
- turn requests-booking into int, so less transforming of strings