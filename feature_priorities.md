# Feature Development

For the basic GLM architecture, we need specific features that we can develop.

## Feature Ideas
- calculate rough maximum time to stretch the maximum operating area distances in meters and calculate to seconds as that is what we use as costs?

- state features (miscellaneous)
    - normalized time over entire request horizon in [0,1]
    - avg. vehicle load (boarded capacities) normalized to [0,1]
    - avg. vehicle promised (active requests) IF KEEP_ACTIVE
    - avg. time until vehicle have finished current route

- vehicle features
    - current capacity load
    - distance of boarded requests (still active, normalized to distance one could do in 30000 - most distance those could have)
    - current position encoding in lat-lon min-max, normalized to [0,1]
    - number of vehicles in vicinity (radius = ???)

- trip features
    - vehicles in vicinity for pickup areas (how can we calculate this based on future trips that still might be in-time?)
    - number of requests, normalized to [0, max_capacity]
    - sum of distances of each separate trip
    - encoding of all locations (how can we do that with a variable number of participating requests)

- vehicle-trip features
    - cost
    - distance to initial pickup, normalized by maximum distance
    - avg. detour distances (as currently visible)
    - avg. waiting time for each request, normalized to total value of maximum waiting time
    - closest value of maxing out a request (time-based) and getting close to infeasibility because a constraint is missed
    - idling time of vehicle (arriving at position before pickup is possible)
    - avg. factor of how much the added trip adds as extra distance (if tour is the same, factor would be -1 as we save 100% of one trip, if they are subsequent, factor 0 (no gain); if the route adds more distance than just the two trips, factor in relation positive)

Assumption: vicinity defined by direct radial distance to a position depsite a road network, reduces calculation of specific network routes (especially useful in the grid-like networks common in the USA which we use for our experiments)

## Future Developments
- CNN for the area where requests are being considered
- overlay of all active trips of vehicles with radius vicinities (not sure how to represent it for NN) - multi-dimensional CNN (based on spatial area 2D and temporal 1D, 1 value for vehicle capacities in that area, 1D for time when one arrives there), looking 20 * step_size into the future would be an "Image" of 2D values in 20 layers with multiple values
- CNN abstraction where all requests have a certain area that they cover around it and a route between stops is also beneficial for that area as cars would move around it
- how can we represent a multi-request trip with many stops in arbitrary order?

Features that Shah et al. 2020 has used:
- vehicle location
- remaining delay of a trip (for boarded requests - 30 min = 1, no minutes left = 0?)
- embeddings for trajecotry with LSTM and two-layer neural network
- information about current decision epoch (??)
- number of vehicles in vicinity
- total number of requests in the epoch
- 