from __future__ import annotations

from typing import Dict, Iterable, List, Tuple, Union
from dataclasses import dataclass, asdict

import numpy as np
from geopy.distance import geodesic

from rtv_solver.structure.trip import Trip
from rtv_solver.structure.shared_trip import SharedTrip
from rtv_solver.structure.trip_cost import TripCost
from rtv_solver.structure.request import Request
from rtv_solver.structure.vehicle import Vehicle
from rtv_solver.handlers.trip_handler import TripHandler
from rtv_solver.structure.config import Config

from rtv_solver.handlers.payload_parser import PayloadParser

FeatureVector = Dict[str, Union[int, float]]


@dataclass
class VehicleFeatures:
    """start with simple 1D scores for a GLM"""
    # TODO alternative for position is a positional embedding of all positions on current trip
    norm_lat_next_position: float = 0.5 # middle of the map, should never be valid
    norm_lon_next_position: float = 0.5 # middle of the map, should never be valid
    # operating_time: float = 0.0 # total operating time
    norm_remaining_operating_period: float = 1.0
    norm_vehicle_count_in_proximity: float = 0.0 # other vehicles in the proximity
    avg_vehicle_distance: float = 0.0 # to all other vehicles
    norm_step_remaining_boarded_time: float = 0.0 # normalized to time per step
    norm_interval_remaining_boarded_time: float = 0.0 # normalized to time per interval
    norm_remaining_am_cap: float = 1.0 # 1 means everything is free
    norm_remaining_wc_cap: float = 1.0 


class FeatureBuilder:
    """
    Simple, extensible feature extraction for RTV states.

    It builds one feature vector per `TripCost` in order to match the size of new scores to the number of feasible trips associated with costs, combining:
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

        self.interval_time = config.BATCH_INTERVAL # based on the iteration steps, this has a different influence
        self.step_time = config.STEP_SIZE
     
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
            fv.update(self._state_features(current_time, vehicles))
            # fv.update(self._trip_features(trip))
            fv.update(self._vehicle_features(vehicle, vehicles, current_time))
            # fv.update(self._trip_cost_features(tc))
            # fv.update(self._request_aggregate_features(trip_requests))

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

    def build_matrix_from_trip_handler(self, trip_handler: TripHandler, current_time):
        trip_costs, trips, vehicles, requests = self._get_components_from_trip_handler(trip_handler)
        return self.build_matrix(trip_costs, trips, vehicles, requests, current_time)
    
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
    
    def _state_features(self, current_time: float, vehicles: list[Vehicle]):
        norm_time = max(0.0, min(1.0, ( current_time - self.r_start_time ) / self.total_operating_time))
        # TODO find a good way to calculate boarded trips at the current_time
        return {
            "norm_time": norm_time
        }

    def _trip_features(self, trip: Union[Trip, SharedTrip]) -> FeatureVector:
        """Basic trip-level features independent of vehicle or request details."""
        if isinstance(trip, Trip):
            cardinality = 1
            base_cost = float(trip.cost) if trip.cost is not None else 0.0
        else:
            cardinality = trip.cardinality
            base_cost = float(trip.cost)

        return {
            "trip_cardinality": cardinality,
            "trip_base_cost": base_cost,
        }

    def _vehicle_features(self, vehicle: Union[Vehicle, None], vehicles: list[Vehicle], current_time: float) -> FeatureVector:
        """Vehicle-related features; returns defaults if vehicle is missing."""
        f = VehicleFeatures()
        BINARY_DISTANCE_CONDITION = 1000 # distance considered close for the spreading of vehicles

        if vehicle is None:
            return asdict(f) # default values
        else:
            if vehicle.trips:
                print("Test")
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
                    print("dist: ", distance_to_vehicle)
                    if distance_to_vehicle < BINARY_DISTANCE_CONDITION: # binary not optimal here
                        vehicle_count_in_proximity += 1.0
                    v_to_v_cum_distance += distance_to_vehicle # bit more neutral than the binary value
            avg_vehicle_distance = v_to_v_cum_distance / (self.total_vehicle_count - 1) / (0.5 * self.max_distance) # only consider half as a vehicle in the corner should have a further distance away (like a circle around it in all directions)

            norm_vehicle_count_in_proximity = vehicle_count_in_proximity / (self.total_vehicle_count - 1)

            am_used, wc_used, am_cap, wc_cap = vehicle.get_capacities()
            remaining_am_cap = (am_cap - am_used) / am_cap
            remaining_wc_cap = (wc_cap - wc_used) / wc_cap
            
            remaining_boarded_time = vehicle.get_remaining_boarded_time(current_time)
            norm_interval_remaining_boarded_time = (remaining_boarded_time - current_time) / self.interval_time
            norm_step_remaining_boarded_time = (remaining_boarded_time-current_time) / self.step_time

            # f.operating_time = veh_operating_end - vehicle.start_time
            f.norm_remaining_operating_period = relative_remaining_operating_period
            f.norm_lat_next_position = norm_lat_position
            f.norm_lon_next_position = norm_lon_position
            f.avg_vehicle_distance = avg_vehicle_distance
            f.norm_vehicle_count_in_proximity = norm_vehicle_count_in_proximity
            f.norm_remaining_am_cap = remaining_am_cap
            f.norm_remaining_wc_cap = remaining_wc_cap
            f.norm_interval_remaining_boarded_time = norm_interval_remaining_boarded_time
            f.norm_step_remaining_boarded_time = norm_step_remaining_boarded_time

            print(vehicle.id, asdict(f))
            return asdict(f)

    def _trip_cost_features(self, trip_cost: TripCost) -> FeatureVector:
        """Features directly derived from TripCost."""
        return {
            "tc_cost": float(trip_cost.cost),
            "tc_sequence_len": len(trip_cost.sequence),
        }

    def _request_aggregate_features(self, requests: List[Request]) -> FeatureVector:
        """
        Aggregate request-related information over all requests in the trip.
        Covers time windows, demand and priority.
        """
        if not requests:
            return {
                "req_count": 0,
                "req_earliest_pickup": 0.0,
                "req_latest_pickup": 0.0,
                "req_earliest_arrival": 0.0,
                "req_latest_arrival": 0.0,
                "req_total_am_demand": 0,
                "req_total_wc_demand": 0,
                "req_avg_priority": 0.0,
            }

        earliest_pickup = min(r.earliest_pickup_time for r in requests)
        latest_pickup = max(r.latest_pickup_time for r in requests)
        earliest_arrival = min(r.earliest_arrival_time for r in requests)
        latest_arrival = max(r.latest_arrival_time for r in requests)
        total_am = sum(r.am_capacity for r in requests)
        total_wc = sum(r.wc_capacity for r in requests)
        avg_priority = sum(r.priority for r in requests) / len(requests)

        return {
            "req_count": len(requests),
            "req_earliest_pickup": earliest_pickup,
            "req_latest_pickup": latest_pickup,
            "req_earliest_arrival": earliest_arrival,
            "req_latest_arrival": latest_arrival,
            "req_total_am_demand": total_am,
            "req_total_wc_demand": total_wc,
            "req_avg_priority": avg_priority,
        }
    
    @staticmethod       
    def _calc_geo_distance_meter(loc1, loc2):
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