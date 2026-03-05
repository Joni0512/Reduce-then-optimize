from __future__ import annotations

from typing import Dict, Iterable, List, Tuple, Union
from dataclasses import dataclass, asdict

import numpy as np
import time
from geopy.distance import geodesic

from rtv_solver.structure.trip import Trip
from rtv_solver.structure.shared_trip import SharedTrip
from rtv_solver.structure.trip_cost import TripCost
from rtv_solver.structure.request import Request
from rtv_solver.structure.vehicle import Vehicle
from rtv_solver.handlers.trip_handler import TripHandler
from rtv_solver.structure.config import Config

from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.structure.trip_insertion_plan import TripInsertionPlan

from rtv_solver.util.logger import BASIC_LOGGER, DATA_LOGGER
import logging

console_logger = logging.getLogger(BASIC_LOGGER)
data_logger = logging.getLogger(DATA_LOGGER)

FeatureVector = Dict[str, Union[int, float]]

"""
using the dataclasses, we can easily add padding with empty items as the default values are already defined"""
# TODO we need a single place where all features are defined and explained and can be easily adjusted in a single place
@dataclass
class StateFeatures:
    """
    basic 1D features for the state and average values across
    
    main value could probably be created through overall vehicle locations encoding or request distributions
    """
    s_norm_time: float = 0.0
    s_avg_remaining_am_cap: float = 1.0
    s_avg_remaining_wc_cap: float = 1.0
    s_total_vehicles: int = 0  # TODO adjust for active vehicles ()
    s_avg_remaining_interval_boarded_time: float = 0.0
    s_avg_remaining_step_boarded_time: float = 0.0


@dataclass
class VehicleFeatures:
    """start with simple 1D scores for a GLM"""
    # TODO alternative for position is a positional embedding of all positions on current trip
    v_norm_lat_next_position: float = 0.5 # middle of the map, should never be valid
    v_norm_lon_next_position: float = 0.5 # middle of the map, should never be valid
    # operating_time: float = 0.0 # total operating time
    v_norm_remaining_operating_period: float = 1.0
    v_norm_vehicle_count_in_proximity: float = 0.0 # other vehicles in the proximity
    v_avg_vehicle_distance: float = 0.0 # to all other vehicles
    v_norm_step_remaining_boarded_time: float = 0.0 # normalized to time per step
    v_norm_interval_remaining_boarded_time: float = 0.0 # normalized to time per interval
    v_norm_remaining_am_cap: float = 1.0 # 1 means everything is free
    v_norm_remaining_wc_cap: float = 1.0
    v_am_cap: int = 0
    v_wc_cap: int = 0 


@dataclass 
class TripCostFeatures:
    """1D feature scores for each request-trip-vehicle combination for aggregated information; that also defines the blank defaults if nothing can be calculated"""
    tc_cost: float = 0.0
    tc_added_cost: float = 0.0
    tc_sequence_len: int = 0
    tc_num_trips: int = 0

    tc_travel_time_to_first_pickup: float = 0.0
    tc_norm_travel_time_to_first_pickup: float = 0.0
    tc_total_direct_travel_time: float = 0.0
    tc_actual_travel_time: float = 0.0
    tc_norm_batch_actual_travel_time: float = 0.0
    tc_total_dwell_time: float = 0.0
    tc_dwell_time_ratio: float = 0.0
    
    tc_detour_time: float = 0.0
    tc_norm_detour_time: float = 0.0
    tc_total_am_demand: int = 0
    tc_total_wc_demand: int = 0

    tc_norm_idling_time: float = 0.0
    tc_sharing_efficiency_factor: float = 0.0

    # tc_actual_route_travel_time: float = 0.0 can be deduced from values above in the features


class FeatureBuilder:
    FEATURE_SIZE = 67 # TODO update this value if you change the features
    REJECT_FLAG_FEATURE_NAME = "action_reject_flag"
    """It builds one feature vector per `TripCost` in order to match the size of new scores to the number of feasible trips associated with costs, combining:
    - vehicle information
    - trip information (single or shared)
    - cost / sequence information
    - aggregated request information
    Future additions might incorporate information of future payloads.

    Implementation shall be deterministic and should use only existing in-memory structures or payload information (minimizing additional network calls).
    """
    def __init__(self, complete_payload: dict, config: Config) -> None:
        """
        Assumptions:
        -----------
          1. As the operating area is quite similar in relation to the surface of the earth, we consider the operating area a perfect square defined by its min-max lat and lon
        """       
        self.config = config
        # initialize the values to normalize (should be usable for most features)
        (self.min_lat, self.max_lat), (self.min_lon, self.max_lon) = PayloadParser.get_request_operating_area_limits(complete_payload)
        self.max_lat_distance = self._calc_geo_distance_meter((self.min_lat, self.max_lon), (self.max_lat, self.max_lon)) # max 'vertical' distance
        self.max_lon_distance = self._calc_geo_distance_meter((self.max_lat, self.min_lon), (self.max_lat, self.max_lon)) # max 'horizontal' distance
        self.max_distance = self._calc_geo_distance_meter((self.min_lat, self.min_lon), (self.max_lat, self.max_lon)) # calculate the maximum distance in meters based on the overall operating area
        
        self.vehicle_operating_intervals = PayloadParser.get_vehicle_time_intervals(complete_payload)
        self.total_vehicle_count = PayloadParser.get_vehicle_count(complete_payload)

        self.total_request_count = PayloadParser.get_request_count(complete_payload)
        self.r_start_time, self.r_end_time = PayloadParser.get_requests_time_interval(complete_payload)
        self.total_operating_time = self.r_end_time - self.r_start_time

        self.BINARY_DISTANCE_CONDITION = 1000

    def build_from_trip_handler(self, trip_handler: TripHandler, current_time: float) -> List[FeatureVector]:
        """
        Build feature dictionaries from a populated TripHandler instance.

        One feature vector is produced per TripCost (i.e. per (trip, vehicle)
        combination that is currently feasible).
        """
        trip_costs, trips, vehicles, requests = self._get_components_from_trip_handler(trip_handler)

        return self.build_from_components(trip_costs, trips, vehicles, requests, current_time)

    def build_from_components(
        self,
        trip_costs: Iterable[TripCost],
        trips: List[Union[Trip, SharedTrip]],
        vehicles: Dict[int, Vehicle],
        requests: List[Request],
        current_time: float
    ) -> List[FeatureVector]:
        """
        Core feature builder that operates on explicit components.

        This is convenient for training scenarios where data comes from
        logs or precomputed snapshots instead of a live TripHandler.
        """
        requests_by_id = {r.id: r for r in requests}
        features: List[FeatureVector] = []

        for tc in trip_costs:
            trip = trips[tc.trip_no]
            vehicle = vehicles.get(tc.vehicle_id)
            trip_request_ids = self._get_request_ids_for_trip(trip, trips)
            trip_requests = [
                requests_by_id[rid] for rid in trip_request_ids if rid in requests_by_id
            ]

            fv: FeatureVector = {}
            fv.update(self._state_features(current_time, vehicles))             # 6 items
            # fv.update(self._trip_features(trip))
            fv.update(self._vehicle_features(vehicle, vehicles, current_time))  # 11 items
            # fv.update(self._trip_cost_features(tc, current_time))               # 17 items
            fv.update(self._future_requests_features(trip_requests, current_time)) # 49 items
            # fv.update(self._request_aggregate_features(trip_requests))
            fv[self.REJECT_FLAG_FEATURE_NAME] = 0.0

            # current total = 6 + 11 + 49 + 1 = 67 items
            # + 11 + 17 + 49 = 83 items

            features.append(fv)

        return features

    def build_matrix(
        self,
        trip_costs: Iterable[TripCost],
        trips: List[Union[Trip, SharedTrip]],
        vehicles: Dict[int, Vehicle],
        requests: List[Request],
        current_time: float
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build a dense feature matrix suitable for ML models.

        Returns:
            matrix: shape (n_trip_costs, n_features)
            feature_names: list of column names in the order used for `matrix`.
        """
        feature_dicts = self.build_from_components(trip_costs, trips, vehicles, requests, current_time)
        if not feature_dicts:
            return np.zeros((0, 0), dtype=float), []

        # Fix column ordering for reproducibility; keep only numeric fields
        example = feature_dicts[0]
        feature_names = sorted(
            name
            for name, value in example.items()
            if isinstance(value, (int, float))
        )
        matrix = np.asarray(
            [[float(fd[name]) for name in feature_names] for fd in feature_dicts],
            dtype=float,
        )
        return matrix, feature_names

    def build_matrix_from_trip_handler(self, trip_handler: TripHandler, current_time: float) -> Tuple[np.ndarray, List[str]]:
        feat_start_time = time.time()
        
        trip_costs, trips, vehicles, requests = self._get_components_from_trip_handler(trip_handler)
        matrix, feature_names = self.build_matrix(trip_costs, trips, vehicles, requests, current_time)
 
        console_logger.info(f"{len(feature_names)} features for {len(trip_costs)} items created in {time.time() - feat_start_time:.3f} s.")

        return matrix, feature_names
    
    def add_reject_action_entries(
        self,
        matrix: np.ndarray,
        feature_names: List[str],
        vehicles: Dict[int, Vehicle],
        current_time: float,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Append one synthetic reject-action row per vehicle.

        For each vehicle, the added row includes:
        - current state features
        - that vehicle's features
        - empty future-request features
        - `action_reject_flag` set to 1

        Returns:
            matrix_with_reject_rows, reject_vehicle_ids
        """
        if matrix.ndim != 2:
            raise ValueError("Feature matrix must be 2-dimensional.")
        if not feature_names:
            raise ValueError("Feature names are required to add reject action entries.")
        if self.REJECT_FLAG_FEATURE_NAME not in feature_names:
            raise ValueError(
                f"Missing reject flag feature '{self.REJECT_FLAG_FEATURE_NAME}' in feature names."
            )
        if matrix.shape[1] != len(feature_names):
            raise ValueError("Feature matrix column count does not match feature names.")

        if not vehicles:
            return matrix, []

        reject_vehicle_ids = sorted(vehicles.keys())
        # Keep deterministic vehicle ordering so appended reject rows map consistently to reject variables/scores across components.
        state_features = self._state_features(current_time, vehicles)
        empty_future_features = self._future_requests_features([], current_time)
        reject_rows: list[list[float]] = []

        for vehicle_id in reject_vehicle_ids:
            row_features: FeatureVector = {}
            row_features.update(state_features)
            row_features.update(self._vehicle_features(vehicles[vehicle_id], vehicles, current_time))
            row_features.update(empty_future_features)
            row_features[self.REJECT_FLAG_FEATURE_NAME] = 1.0

            reject_rows.append(
                # Project by feature_names to guarantee exact column alignment with the existing feature matrix.
                [float(row_features.get(name, 0.0)) for name in feature_names]
            )

        reject_array = np.asarray(
            reject_rows, dtype=matrix.dtype if matrix.size > 0 else float
        )
        matrix_with_reject = np.vstack([matrix, reject_array])
        return matrix_with_reject, reject_vehicle_ids
        
    
    # INTERNAL HELPERS
    def _get_components_from_trip_handler(self, trip_handler: TripHandler):
        vehicles = trip_handler.vehicles
        requests = trip_handler.requests
        trips: List[Union[Trip, SharedTrip]] = trip_handler.trips
        trip_costs: Iterable[TripCost] = getattr(TripHandler, "trip_costs", [])

        return trip_costs, trips, vehicles, requests

    def _get_request_ids_for_trip(
        self, trip: Union[Trip, SharedTrip], trips: List[Union[Trip, SharedTrip]]
    ) -> List[int]:
        """Return the request ids that belong to this trip (handles shared trips)."""
        if isinstance(trip, Trip):
            return [trip.request_id]

        # SharedTrip: resolve to underlying single-request Trips
        request_ids: List[int] = []
        for sub_trip_no in trip.trips:
            sub_trip = trips[sub_trip_no]
            if isinstance(sub_trip, Trip):
                request_ids.append(sub_trip.request_id)
        return request_ids
    
    def _state_features(self, current_time: float, vehicles: dict[int, Vehicle]):
        norm_time = max(0.0, min(1.0, ( current_time - self.r_start_time ) / self.total_operating_time))
        
        total_remaining_am_caps = 0.0
        total_remaining_wc_caps = 0.0
        total_norm_interval_remaining_boarded_time = 0.0
        total_norm_step_remaining_boarded_time = 0.0
        for vid, vehicle in vehicles.items():
            _, _, remaining_am_cap, remaining_wc_cap = vehicle.get_remaining_capacities()
            total_remaining_am_caps += remaining_am_cap
            total_remaining_wc_caps += remaining_wc_cap

            remaining_boarded_time = vehicle.get_remaining_boarded_time(current_time)
            norm_interval_remaining_boarded_time = max(0.0, (remaining_boarded_time - current_time) / self.config.BATCH_INTERVAL)
            norm_step_remaining_boarded_time = max(0.0, (remaining_boarded_time - current_time) / self.config.STEP_SIZE)
            total_norm_interval_remaining_boarded_time += norm_interval_remaining_boarded_time
            total_norm_step_remaining_boarded_time += norm_step_remaining_boarded_time
        
        avg_am_cap = total_remaining_am_caps / self.total_vehicle_count
        avg_wc_cap = total_remaining_wc_caps / self.total_vehicle_count

        avg_interval_boarded_time = total_norm_interval_remaining_boarded_time / self.total_vehicle_count
        avg_step_boarded_time = total_norm_step_remaining_boarded_time / self.total_vehicle_count    

        sf = StateFeatures()
        sf.s_norm_time = norm_time
        sf.s_avg_remaining_am_cap = avg_am_cap
        sf.s_avg_remaining_wc_cap = avg_wc_cap
        sf.s_total_vehicles = self.total_vehicle_count
        sf.s_avg_remaining_interval_boarded_time = avg_interval_boarded_time
        sf.s_avg_remaining_step_boarded_time =  avg_step_boarded_time

        return asdict(sf)

    def _trip_features(self, trip: Union[Trip, SharedTrip]) -> FeatureVector:
        """Basic trip-level features independent of vehicle or request details."""
        if isinstance(trip, Trip):
            cardinality = 1
            base_cost = float(trip.cost) if trip.cost is not None else 0.0
        else: # SharedTrip
            cardinality = trip.cardinality
            base_cost = float(trip.cost)

        return {
            "trip_cardinality": cardinality,
            "trip_base_cost": base_cost,
        }

    def _vehicle_features(self, vehicle: Union[Vehicle, None], vehicles: dict[int, Vehicle], current_time: float) -> FeatureVector:
        """Vehicle-related features; returns defaults if vehicle is missing."""
        # TODO clean this up to add new items more easily and add descriptions that are also required for the thesis
        f = VehicleFeatures()
         # distance considered close for the spreading of vehicles

        if vehicle is None:
            return asdict(f) # default values
        else:
            default_vertical = (self.min_lat, vehicle.next_immediate_node.lon), 
            default_horizontal = (vehicle.next_immediate_node.lat, self.min_lon)
            vehicle_pos = (vehicle.next_immediate_node.lat, vehicle.next_immediate_node.lon)
            norm_lat_position = self._calc_geo_distance_meter(default_vertical, vehicle_pos) / self.max_lat_distance
            norm_lon_position = self._calc_geo_distance_meter(default_horizontal, vehicle_pos) / self.max_lon_distance

            veh_operating_end = min(vehicle.end_time, self.r_end_time)
            if current_time > vehicle.start_time and vehicle.started:
                relative_remaining_operating_period = max(0.0, min((veh_operating_end - current_time) / (veh_operating_end - vehicle.start_time), 1.0))
            else:
                relative_remaining_operating_period = 1.0

            vehicle_count_in_proximity = 0.0 # should adjust for active vehicles?#
            v_to_v_cum_distance = 0.0
            for sep_veh_id, sep_vehicle in vehicles.items():
                if sep_veh_id != vehicle.id:
                    sep_position = (sep_vehicle.next_immediate_node.lat, sep_vehicle.next_immediate_node.lon)
                    distance_to_vehicle = self._calc_geo_distance_meter(sep_position, vehicle_pos)
                    if distance_to_vehicle < self.BINARY_DISTANCE_CONDITION: # binary not optimal here
                        vehicle_count_in_proximity += 1.0
                    v_to_v_cum_distance += distance_to_vehicle # bit more neutral than the binary value
            if self.total_vehicle_count - 1 > 0:
                avg_vehicle_distance = v_to_v_cum_distance / (self.total_vehicle_count - 1) / (0.5 * self.max_distance) # only consider half as a vehicle in the corner should have a further distance away (like a circle around it in all directions)
                norm_vehicle_count_in_proximity = vehicle_count_in_proximity / (self.total_vehicle_count - 1)
            else:
                avg_vehicle_distance = 0.0
                norm_vehicle_count_in_proximity = 0.0

            am_cap, wc_cap, norm_remaining_am_cap, norm_remaining_wc_cap = vehicle.get_remaining_capacities()

            remaining_boarded_time = vehicle.get_remaining_boarded_time(current_time)
            norm_interval_remaining_boarded_time = max(0.0, (remaining_boarded_time - current_time) / self.config.BATCH_INTERVAL)
            norm_step_remaining_boarded_time = max(0.0, (remaining_boarded_time - current_time) / self.config.STEP_SIZE)

            # f.operating_time = veh_operating_end - vehicle.start_time
            f.v_norm_remaining_operating_period = relative_remaining_operating_period
            f.v_norm_lat_next_position = norm_lat_position
            f.v_norm_lon_next_position = norm_lon_position
            f.v_avg_vehicle_distance = avg_vehicle_distance
            f.v_norm_vehicle_count_in_proximity = norm_vehicle_count_in_proximity
            f.v_norm_remaining_am_cap = norm_remaining_am_cap
            f.v_norm_remaining_wc_cap = norm_remaining_wc_cap
            f.v_am_cap = am_cap
            f.v_wc_cap = wc_cap
            f.v_norm_interval_remaining_boarded_time = norm_interval_remaining_boarded_time
            f.v_norm_step_remaining_boarded_time = norm_step_remaining_boarded_time

            return asdict(f)

    def _trip_cost_features(self, trip_cost: TripCost, current_time: float) -> FeatureVector:
        """
        Calculates features from TripCost/TripInsertionPlan data and returns them as a flat dict (FeatureVector).
        All feature defaults are defined in TripCostFeatures.

        Feature categories:
        1. Cost - total insertion cost and added marginal cost; sequence length and number of trips.
        2. Travel time to first pickup - raw value and normalized by max_distance (time units).
        3. Detour time - min/max individual direct trip times, total direct travel time, actual
           route travel time (normalized by BATCH_INTERVAL), total dwell time and dwell-time ratio
           relative to actual route time, and absolute/normalized detour time.
        4. Idling time - time the vehicle waits at a stop before the earliest pickup is allowed,
           normalized by BATCH_INTERVAL.
        5. Sharing efficiency factor - (actual_travel_time - total_direct_travel_time) /
           total_direct_travel_time; positive means extra travel overhead, would be negative for savings.
        6. Capacity demand - total ambulatory (am) and wheelchair (wc) demand across all trips in the plan.

        NOTE: All travel-time values are pre-calculated in TripInsertionPlan during
        vehicle_handler.plan_trip_insertions() to avoid redundant network calls.
        """
        # TODO check accuracy of results and build test problem that checks this
        plan: TripInsertionPlan = trip_cost.plan
        features: TripCostFeatures = TripCostFeatures()
        
        # COST
        features.tc_cost = float(trip_cost.cost)
        features.tc_sequence_len = len(plan.sequence)
        features.tc_num_trips = len(plan.trips)
        if plan.added_cost >= 0:
            features.tc_added_cost = float(plan.added_cost)

        # TRAVEL TIME TO FIRST PICKUP
        # Use veh_travel_time which is pre-calculated in vehicle_handler, meaning for current time to get to the next stop
        if plan.veh_travel_time is not None:
            features.tc_travel_time_to_first_pickup = float(plan.veh_travel_time)
            max_travel_time = self.max_distance  # Cost is in time units
            features.tc_norm_travel_time_to_first_pickup = (plan.veh_travel_time / max_travel_time)
        
        # DETOUR TIME        
        # Total and average direct travel times
        if plan.total_direct_travel_time is not None:
            features.tc_total_direct_travel_time = float(plan.total_direct_travel_time)        
        # Actual route breakdown: travel vs dwell
        if plan.actual_travel_time is not None:
            features.tc_actual_travel_time = float(plan.actual_travel_time)
            features.tc_norm_batch_actual_travel_time = features.tc_actual_travel_time / self.config.BATCH_INTERVAL
        if plan.total_dwell_time is not None:
            features.tc_total_dwell_time = float(plan.total_dwell_time)
            # Dwell time ratio relative to total route time
            if plan.actual_route_travel_time and plan.actual_route_travel_time > 0:
                features.tc_dwell_time_ratio = (
                    plan.total_dwell_time / plan.actual_travel_time)
        # Detour metrics
        if plan.detour_time is not None:
            features.tc_detour_time = float(plan.detour_time)
            features.tc_norm_detour_time = plan.detour_time / self.max_distance

        # IDLING TIME
        # Time vehicle waits after arriving before pickup is allowed
        if plan.idling_time is not None:
            features.tc_norm_idling_time = plan.idling_time / self.config.BATCH_INTERVAL

        # SHARING EFFICIENCY FACTOR
        # Factor of how much extra travel time is added (negative = savings)
        if (plan.total_direct_travel_time is not None and 
            plan.actual_travel_time is not None and 
            plan.total_direct_travel_time > 0):
            
            total_direct = plan.total_direct_travel_time
            actual_route = plan.actual_travel_time # not consider default dwell times

            # Efficiency: (actual - direct) / direct
            # Negative = (savings), positive = (extra travel time)
            sharing_factor = (actual_route - total_direct) / total_direct
            features.tc_sharing_efficiency_factor = sharing_factor
        
        # CAPACITY FEATURES
        total_am_demand = sum(trip.am_capacity for trip in plan.trips)
        total_wc_demand = sum(trip.wc_capacity for trip in plan.trips)
        features.tc_total_am_demand = float(total_am_demand)
        features.tc_total_wc_demand = float(total_wc_demand)

        # TODO add remaining times in relation to the current_time for current dropoffs (basically potential for later detours)

        # print(trip_cost.trip_no, trip_cost.vehicle_id, features)
        return asdict(features)

    def _future_requests_features(self, requests: List[Request], current_time: float) -> FeatureVector:
        """
        Aggregates future request pickup demand onto a 7x7 geographic grid with interval-based
        temporal decay.

        The operating area (defined by min/max lat and lon) is divided into a 7x7 grid.
        For each request whose earliest pickup time is in the future, the pickup origin is
        mapped to a grid cell and a decay score is accumulated:

          - Interval 1  [current_time,                current_time + 1*BATCH_INTERVAL): score = 1.0
          - Interval 2  [current_time + BATCH_INTERVAL,  current_time + 2*BATCH_INTERVAL): score = 0.5
          - Interval 3  [current_time + 2*BATCH_INTERVAL, current_time + 3*BATCH_INTERVAL): score = 0.25
          - Interval 4  [current_time + 3*BATCH_INTERVAL, current_time + 4*BATCH_INTERVAL): score = 0.125
          - Beyond interval 4: ignored

        Requests whose earliest_pickup_time falls beyond the look-ahead horizon
        (4 * BATCH_INTERVAL) are ignored. Dropoff location and capacity are not used.

        Returns a 49-element FeatureVector keyed as ``fr_grid_{row}_{col}`` where row 0 / col 0
        corresponds to the south-west corner of the operating area (min lat / min lon).

        Parameters
        ----------
        requests : List[Request]
            Candidate future requests (not yet dispatched). Only requests with
            earliest_pickup_time strictly after current_time are considered.
        current_time : float
            Current simulation time.

        Returns
        -------
        FeatureVector
            Flat dict with 49 entries (7×7 grid), each value being the sum of decay scores
            accumulated for that grid cell.
        """
        GRID_SIZE = 7
        NUM_INTERVALS = 4

        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)

        lat_range = self.max_lat - self.min_lat
        lon_range = self.max_lon - self.min_lon

        for request in requests:
            pickup_time = request.earliest_pickup_time
            if pickup_time <= current_time:
                continue

            # Which 1-indexed interval does this pickup fall into?
            offset = pickup_time - current_time
            interval_index = int(offset // self.config.BATCH_INTERVAL) + 1

            if interval_index > NUM_INTERVALS:
                continue

            decay = 0.5 ** (interval_index - 1)

            # Map pickup lat/lon to grid cell (row=south→north, col=west→east)
            if lat_range > 0:
                row = int((request.origin.lat - self.min_lat) / lat_range * GRID_SIZE)
            else:
                row = 0
            if lon_range > 0:
                col = int((request.origin.lon - self.min_lon) / lon_range * GRID_SIZE)
            else:
                col = 0

            row = min(row, GRID_SIZE - 1)
            col = min(col, GRID_SIZE - 1)

            grid[row, col] += decay

        return {
            f"fr_grid_{row}_{col}": float(grid[row, col])
            for row in range(GRID_SIZE)
            for col in range(GRID_SIZE)
        }
    
    @staticmethod       
    def _calc_geo_distance_meter(loc1: tuple[float, float], loc2: tuple[float, float]):
        """each location must be defined as a tuple with (lat, lon)"""
        return geodesic(loc1, loc2).meters
    
    def to_dict(self):
        return {
            "max_lat_distance": self.max_lat_distance,
            "max_lon_distance": self.max_lon_distance,
            "max_distance": self.max_distance,
            "total_vehicle_count": self.total_vehicle_count,
            "total_operating_time": self.total_operating_time,
            "r_start_time": self.r_start_time,
            "r_end_time": self.r_end_time
        }