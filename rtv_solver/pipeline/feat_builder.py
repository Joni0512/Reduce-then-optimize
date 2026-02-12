from __future__ import annotations

from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
from geopy.distance import geodesic

from rtv_solver.structure.trip import Trip
from rtv_solver.structure.shared_trip import SharedTrip
from rtv_solver.structure.trip_cost import TripCost
from rtv_solver.structure.request import Request
from rtv_solver.structure.vehicle import Vehicle
from rtv_solver.handlers.trip_handler import TripHandler

from rtv_solver.handlers.payload_parser import PayloadParser

FeatureVector = Dict[str, Union[int, float]]


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
    def __init__(self, complete_payload) -> None:
        # initialize the values to normalize (should be usable for most features)
        self.min_lat, self.max_lat, self.min_lon, self.max_lon = PayloadParser.get_request_operating_area_limits(complete_payload)
        self.lat_distance = self._calc_geo_distance_meter((self.min_lat, self.max_lon), (self.min_lat, self.max_lon)) # max 'vertical' distance
        self.lon_distance = self._calc_geo_distance_meter((self.max_lat, self.min_lon), (self.max_lat, self.max_lon)) # max 'horizontal' distance
        self.max_distance = self._calc_geo_distance_meter((self.min_lat, self.max_lat), (self.min_lon, self.max_lon)) # calculate the maximum distance in meters based on the overall operating area
        
        self.vehicle_operating_intervals = PayloadParser.get_vehicle_time_intervals(complete_payload)
        self.start_time, self.end_time = PayloadParser.get_requests_time_interval(complete_payload)
        self.total_operating_time = self.end_time - self.start_time

        # TODO add fixed normalization values and sort features
    def build_state_features(self):
        pass 
     
    def build_from_trip_handler(self, trip_handler: TripHandler) -> List[FeatureVector]:
        """
        Build feature dictionaries from a populated TripHandler instance.

        One feature vector is produced per TripCost (i.e. per (trip, vehicle)
        combination that is currently feasible).
        """
        trip_costs, trips, vehicles, requests = self._get_components_from_trip_handler(trip_handler)

        return self.build_from_components(trip_costs, trips, vehicles, requests)

    def build_from_components(
        self,
        trip_costs: Iterable[TripCost],
        trips: List[Union[Trip, SharedTrip]],
        vehicles: Dict[int, Vehicle],
        requests: List[Request],
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
            fv.update(self._trip_features(trip))
            fv.update(self._vehicle_features(vehicle))
            fv.update(self._trip_cost_features(tc))
            fv.update(self._request_aggregate_features(trip_requests))

            # Optional identity features to allow joining back to structures
            fv["trip_no"] = tc.trip_no
            fv["vehicle_id"] = tc.vehicle_id

            features.append(fv)

        return features

    def build_matrix(
        self,
        trip_costs: Iterable[TripCost],
        trips: List[Union[Trip, SharedTrip]],
        vehicles: Dict[int, Vehicle],
        requests: List[Request],
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build a dense feature matrix suitable for ML models.

        Returns:
            matrix: shape (n_trip_costs, n_features)
            feature_names: list of column names in the order used for `matrix`.
        """
        feature_dicts = self.build_from_components(trip_costs, trips, vehicles, requests)
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

    def build_matrix_from_trip_handler(self, trip_handler: TripHandler):
        trip_costs, trips, vehicles, requests = self._get_components_from_trip_handler(trip_handler)
        return self.build_matrix(trip_costs, trips, vehicles, requests)
    
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
        norm_time = ( current_time - self.start_time ) / self.total_operating_time
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

    def _vehicle_features(self, vehicle: Union[Vehicle, None]) -> FeatureVector:
        """Static vehicle-related features; returns zeros if vehicle is missing."""
        if vehicle is None:
            return {
                "veh_am_capacity": 0,
                "veh_wc_capacity": 0,
            }
        # what is the actual location in-between
        lat_pos = vehicle.next_immediate_node

        return {
            "veh_am_capacity": vehicle.am_capacity,
            "veh_wc_capacity": vehicle.wc_capacity,
        }

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

if __name__ == '__main__':
    from rtv_solver.tests.conftest import trip

    print(repr(trip))

