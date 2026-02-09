# System overview

The codebased implements both an online and an offline request-trip-vehicle (RTV) solver with an Integer Linear Programme (ILP) solving a Pickup and Delivery Problem with Time Windows (PDPTW). All core software components are organised in the `rtv_solver` package. 
The codebase is used for research purposes and experimental runs with different data and config settings.
The codebase is used to implement a new approach integrating machine learning (ML) or specifically structured reinforcement learning (SRL) to improve the assignment of the RTV approach in long-term horizons.

Execution entry points:
- Code execution via `rtv_solver/main_ors.py`. 
- CLI / Batch execution via shell script `rtv_solver/run_main_ors.sh`

The implementation covers the full solver pipeline, including payload parsing, trip generation, feasibility checks, optimization via Gurobi, and result output.

The script `rtv_solver/main_ors.py` handles the argument parsing to guarantee reproducible experiments, collects the input data and instantiates logging.

The script either calls the online solver in `rtv_solver/online_rtv_solver.py` that solves the entire payload that was given to it. 
As an alternative, the script calls the offline solver in `rtv_solver/offline_rtv_solver.py` that generates a solution by splitting up payload batches and incrementing the simulation time of the vehicles independently. 

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

# Agent instructions

You are working in a legacy codebase.

The following constraints are critical:
- Backward compatibility is mandatory
- Avoid refactoring unless explicitly requested
- Highlight any side effects you introduce
- Prefer minimal diffs and localized changes
- Do not change public APIs without approval
- Do not build parallel multiprocessing by default, rather add a note but build a single-thread and single-process approach for testing.

Assume that behavior is historically and functionally motivated even when it appears suboptimal.

## Before making code changes

Before writing or modifying code:
1. Identify which files will be changed and why
2. State which invariants might be affected
3. Explicitly confirm that backward compatibility is preserved
4. Call out any uncertainty or assumptions

Only then propose code changes.

## Change Risk Levels

Low-risk (generally safe to modify):
- `rtv_solver/visuals/`
- `rtv_solver/util/`
- Tests under `rtv_solver/tests/`

Medium-risk:
- `payload_parser.py`
- `stats_parser.py`

High-risk / Danger zone:
- `trip_handler.py`
- `vehicle_handler.py`
- `network_handler.py`
- ILP formulation and Gurobi calls

Changes in high-risk modules require extra justification and minimal diffs.

## Side Effects (Must Be Explicitly Mentioned)

The following count as side effects and must be called out:
- Mutation of class-level or module-level state
- Changes to solver ordering or execution timing
- Additional network calls (OSRM)
- Changes to ILP constraints, objectives, or solver parameters
- Changes to logging behavior or output formats

## Validation Expectations

- Existing tests must continue to pass unchanged or only with minor changes of the API.
- If you add new methods and classes, add tests that check the API and ensure working code. 
- Keep the structure of existing tests.

# Project-specific conventions and pitfalls

- `rtv_solver/main_ors.py` 
  CLI entry point, batch loop, example flags (see `if __name__ == "__main__"`).

- `rtv_solver/handlers/trip_handler.py` 
  Contains extensive logic for:
  - Trip generation
  - Shared-trip combinations
  - Attempted multiprocessing / parallelization  
  - ATTENTION This is one of the most fragile and complex modules.
  - Several class-level or module-level globals are used as shared state, e.g.:
    - `TripHandler.trip_costs`
    - `TripHandler.shared_trips_to_create`
    - These globals are designed for single-process execution. 
    - **Worker processes must NOT directly mutate these globals.**

- **`rtv_solver/handlers/network_handler.py`**  
  - Wrapper around OSRM / travel-time matrix.  
  - Important for understanding initialization, I/O patterns, and multiprocessing behavior.
  - `NetworkHandler.init(...)` is treated as a **global initialization step** to handle all nodes in a shared handler.

- keep state mutation in the main process
  - workers should return only serializable results (e.g., primitives, tuples, and dicts)
  - in the main state, convert results into data objects for easier handling in the main process