# Feature Development

For the basic GLM architecture, we need specific features that we can develop.

## Feature Ideas
- calculate rough maximum time to stretch the maximum operating area distances in meters and calculate to seconds as that is what we use as costs?

- state features (miscellaneous)
    - DONE normalized time over entire request horizon in [0,1]
    - DONE avg. vehicle load (boarded capacities) normalized to [0,1] (for am and wc)
    - (less prio) avg. vehicle promised (active requests) IF KEEP_ACTIVE
    - DONE avg. time until active vehicles have finished current route with all boarded requests
    - active vehicles (really pay attention to 'active') >> P2 as all vehicles have the same active time for now
    - DONE total_vehicle_count
    - number of decisions required in interval / step

- vehicle features
    - DONE current capacity load (boarded trips)
    - DONE time to boarded requests (still active, normalized to times of steps and intervals)
    - DONE current position encoding in lat-lon min-max, normalized to [0,1], what is 
    - DONE number of vehicles in vicinity (radius = 1000, defined in the function)
    - DONE avg. distance to all other vehicles
    - DONE total operating time
    - DONE relative part of operating time fulfilled (relative to max operating time of that vehicle)
    - DONE relative remaining capacities (for am and wc)
    - DONE total capacities (for am and wc) (to enable different vehicles)
    - remaining active time, so last stop already registered in manifest (applies if we keep_active so state t and t+1 are more closely correlated)

- trip features
    - (DONE) sum of distances of each separate trip (relative to real cost - single trip would be 1.0, shared trip should be lower as a pre-condition for higher service rate)
    - DONE total distance (cost of trip)
    - remaining wait time of a trip, how many intervals can we postpone? (nudge towards servicing earlier requests as more important)
    - DONE number of requests, normalized to [0, max_cardinality], what is the highest possible value
    - vehicles in vicinity for pickup areas (how can we calculate this based on future trips that still might be in-time?)

- vehicle-trip / trip cost features
    - DONE cost
    - DONE distance to initial pickup, normalized by maximum distance
    - (done) avg. detour distances (how much distance do the direct trips have and how much distance are we adding to connect trips, negative value is more positive?)
    - (done) avg. waiting time for each request, normalized to total value of maximum waiting time
    - closest value of maxxing out a request (time-based) and getting close to infeasibility because a constraint is missed
    - (done) avg. idling time of vehicle (arriving at position before pickup is possible)
    - DONE avg. factor of how much the added trip adds as extra distance (if tour is the same, factor would be -1 as we save 100% of one trip, if they are subsequent, factor 0 (no gain); if the route adds more distance than just the two trips, factor in relation positive)

- future payload information
    - split area in grid cells (7x7) and add decay factor with more value for recent timestep (first interval +1, second interval +0.5, ...)

shared trip features / requests aggregated?
    - encoding of all locations (how can we do that with a variable number of participating requests)

Assumption: vicinity defined by direct radial distance to a position depsite a road network, reduces calculation of specific network routes (especially useful in the grid-like networks common in the USA which we use for our experiments)

## Future Developments
- CNN for the area where requests are being considered
- overlay of all active trips of vehicles with radius vicinities (not sure how to represent it for NN) - multi-dimensional CNN (based on spatial area 2D and temporal 1D, 1 value for vehicle capacities in that area, 1D for time when one arrives there), looking 20 * step_size into the future would be an "Image" of 2D values in 20 layers with multiple values
- CNN abstraction where all requests have a certain area that they cover around it and a route between stops is also beneficial for that area as cars would move around it
- how can we represent a multi-request trip with many stops in arbitrary order?
- vehicles soon-to-be-finishing

Features that Shah et al. 2020 has used:
- vehicle location
- remaining delay of a trip (for boarded requests - 30 min = 1, no minutes left = 0?)
- embeddings for trajecotry with LSTM and two-layer neural network
- information about current decision epoch (??)
- number of vehicles in vicinity
- total number of requests in the epoch
- 