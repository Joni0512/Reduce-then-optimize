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
#das wurde geändert vorher stand es untereinander

#from rtv_solver.handlers.trip_handler import TripHandler
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
# currently it is not very well designed to make the feature set adaptable, maintainable or even testable; if we want to change anything it is quite cumbersome. check online what a good design would look like especially in relatio
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
    # TODO alternative for position is a positional embedding (with CNN or something similar)of all positions on current trip
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
class TripFeatures:
    """
    Trip-level composition features that depend on trip structure and the
    associated vehicle capacity context.
    """
    t_num_requests_in_trip: float = 0.0
    t_norm_num_requests_by_vehicle_max_capacity: float = 0.0


@dataclass
class CandidateRequestFeatures:
    """
    2026-07-30: compact candidate-specific counterpart to the (now genuinely
    global) future-demand grid - see _candidate_request_location_features and
    _global_future_demand_features. Previously the 49-cell grid was computed
    over just this candidate's own 1-2 requests, which is almost always
    sparse/near-empty and largely redundant with tc_travel_time_to_first_pickup
    (see feature-group ablation in outputs/ablation_bi200_ss100_class1_legacy -
    the grid contributed ~0 to service rate there, most likely because of this).
    This replaces that with the actual non-redundant signal a per-candidate
    grid cell was trying to provide: where this candidate's own request(s) are,
    and how urgent the most pressing one is.
    """
    cr_norm_lat: float = 0.5  # mean pickup latitude of this candidate's own requests; 0.5 = no requests / center of map (never a valid real position, same convention as v_norm_lat_next_position)
    cr_norm_lon: float = 0.5
    cr_urgency: float = 0.0  # max decay score (see _global_future_demand_features) across this candidate's own requests; 1.0 = due now, 0.0 = no requests or beyond look-ahead horizon
    # 2026-08-09: continuous counterpart to cr_urgency - cr_urgency is binned/decayed
    # against earliest_pickup_time (window OPEN), this is the raw, unbinned distance
    # to latest_pickup_time (the HARD deadline), same formula match_graph_features.py
    # already uses for the SRL critic (pickup_slack there). Min across the candidate's
    # own requests = the most urgent one drives the risk (mirrors cr_urgency's "max
    # decay = most urgent" convention). Default 4.0 matches the look-ahead horizon
    # cutoff cr_urgency already uses (interval_index > 4), i.e. "no pressing deadline".
    cr_pickup_slack: float = 4.0


@dataclass
class CompetitionFeatures:
    """
    2026-08-09: second wave of v2 additions (on top of the global-grid/cr_*/
    reject-fix changes from 2026-07-30) - how competitive is this candidate
    compared with the alternatives that could replace it? Right now every
    candidate row is scored by the MLP in isolation; these give it a cheap
    signal about its local competition without needing message passing (the
    GNN already sees this implicitly via CandidateConflictGraphBuilder edges).
    Toggle with FeatureBuilder.ENABLE_COMPETITION_FEATURES - same on/off
    pattern as ENABLE_TRIP_COMPOSITION_FEATURES - so baseline (v1) / v2 /
    v2+competition stay comparable via one flag instead of a third file.
    """
    cf_cost_minus_best_for_vehicle: float = 0.0  # 0.0 = cheapest option for this vehicle
    cf_cost_ratio_best_for_request: float = 1.0  # 1.0 = matches the best cost available anywhere for this request
    cf_norm_rank_for_vehicle: float = 0.0  # 0 = cheapest among this vehicle's alternatives, towards 1 = worst
    cf_num_candidates_same_vehicle: int = 1  # includes self
    cf_num_candidates_same_request: int = 1  # includes self; union across this candidate's own request(s)
    cf_diff_to_mean_competing_cost: float = 0.0  # 0.0 if no other candidates compete for this vehicle
    cf_conflict_graph_degree: int = 0  # # other candidates sharing this vehicle or any of this candidate's requests


@dataclass
class TripCostFeatures:
    """1D feature scores for each request-trip-vehicle combination for aggregated information; that also defines the blank defaults if nothing can be calculated"""
    # tc_cost: float = 0.0
    # tc_added_cost: float = 0.0
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
    ENABLE_TRIP_COMPOSITION_FEATURES = True # include explicit trip-composition signal
    # 2026-08-09: second wave of v2 - see CompetitionFeatures. Default True; set
    # False to reproduce the original (first-wave) v2 feature set for comparison.
    ENABLE_COMPETITION_FEATURES = True
    # 2026-08-09: cr_pickup_slack (continuous distance to the hard pickup deadline,
    # see CandidateRequestFeatures) - toggle so the pre-pickup_slack v2 feature set
    # stays reproducible (False) for comparison against the already-running
    # fbv2_seed*/fbv2comp_seed* cluster jobs, which used feat_builder_new.py before
    # this field existed.
    ENABLE_PICKUP_SLACK_FEATURE = True
    # 2026-07-30: base = 6 (state) + 11 (vehicle) + 15 (tripcost) + 49 (global future
    # demand grid) + 3 (candidate request location) + 1 (reject flag) = 85.
    # (Previously documented as 82 = 6+11+"17"(actually 15)+49+1 - the "17" was a
    # stale comment in build_from_components, not a real extra 2 features.)
    _BASE_FEATURE_SIZE = 85
    _TRIP_COMPOSITION_FEATURE_SIZE = 2
    # 2026-08-09: cf_cost_minus_best_for_vehicle, cf_cost_ratio_best_for_request,
    # cf_norm_rank_for_vehicle, cf_num_candidates_same_vehicle,
    # cf_num_candidates_same_request, cf_diff_to_mean_competing_cost,
    # cf_conflict_graph_degree - see CompetitionFeatures.
    _COMPETITION_FEATURE_SIZE = 7
    _PICKUP_SLACK_FEATURE_SIZE = 1
    FEATURE_SIZE = _BASE_FEATURE_SIZE + (
        _TRIP_COMPOSITION_FEATURE_SIZE if ENABLE_TRIP_COMPOSITION_FEATURES else 0
    ) + (
        _COMPETITION_FEATURE_SIZE if ENABLE_COMPETITION_FEATURES else 0
    ) + (
        _PICKUP_SLACK_FEATURE_SIZE if ENABLE_PICKUP_SLACK_FEATURE else 0
    )  # TODO update this value if you change the features
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

        # 2026-07-30: computed once per call, not per trip_cost row - this only
        # depends on the batch's full request pool and current_time, identical
        # for every row (same as state_features, which is still recomputed per
        # row below - left as-is to keep this change scoped to the future-demand
        # split). See _global_future_demand_features for why this now uses the
        # full `requests` list instead of one candidate's own requests.
        global_future_demand = self._global_future_demand_features(requests, current_time)

        # 2026-08-09: materialize once - _build_competition_lookup() needs to iterate
        # trip_costs fully before the main per-row loop below also iterates it, and the
        # caller-supplied `trip_costs` is only typed as Iterable (could be a generator).
        trip_costs = list(trip_costs)
        competition_lookup = (
            self._build_competition_lookup(trip_costs)
            if self.ENABLE_COMPETITION_FEATURES
            else None
        )

        for tc in trip_costs:
            trip = trips[tc.trip_no]
            vehicle = vehicles.get(tc.vehicle_id)
            trip_request_ids = self._get_request_ids_for_trip(trip, trips)
            trip_requests = [
                requests_by_id[rid] for rid in trip_request_ids if rid in requests_by_id
            ]

            fv: FeatureVector = {}
            fv.update(self._state_features(current_time, vehicles))             # 6 items
            if self.ENABLE_TRIP_COMPOSITION_FEATURES:
                fv.update(self._trip_features(trip, vehicle))                   # 2 items
            fv.update(self._vehicle_features(vehicle, vehicles, current_time))  # 11 items
            fv.update(self._trip_cost_features(tc, current_time))               # 15 items
            fv.update(global_future_demand)                                     # 49 items - global, same for every row this call
            cr_features = self._candidate_request_location_features(trip_requests, current_time)
            if not self.ENABLE_PICKUP_SLACK_FEATURE:
                cr_features.pop("cr_pickup_slack", None)
            fv.update(cr_features)  # 3 items (4 with pickup slack) - this candidate's own requests
            if self.ENABLE_COMPETITION_FEATURES:
                by_vehicle, by_request = competition_lookup
                fv.update(self._competition_features(tc, by_vehicle, by_request))  # 7 items
            # fv.update(self._request_aggregate_features(trip_requests))
            fv[self.REJECT_FLAG_FEATURE_NAME] = 0.0

            # current total = 85 items (or 87 with trip composition, +7 more with
            # competition, +1 more with pickup slack -> 95 with all three enabled)

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
        requests: List[Request] = (),
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Append one synthetic reject-action row per vehicle.

        For each vehicle, the added row includes:
        - current state features
        - that vehicle's features
        - the same global future-demand grid every trip-candidate row for this
          batch gets (rejecting a candidate doesn't change what else is
          pending, so this should carry the real signal, not zeros)
        - candidate-request-location features at their "no candidate" default
          (no request is actually attached to a reject action)
        - `action_reject_flag` set to 1

        2026-07-30: `requests` is the same full batch request list passed to
        build_from_components/build_matrix - needed now that future-demand is
        computed globally rather than per (empty) candidate. Defaults to ()
        for backward compatibility with existing call sites; passing nothing
        just means reject rows fall back to an all-zero global-demand grid
        (the old behavior) until callers are updated to pass it through.

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
        global_future_demand = self._global_future_demand_features(list(requests), current_time)
        no_candidate_location = self._candidate_request_location_features([], current_time)
        # 2026-08-09: a reject action isn't a real trip candidate, so it has no
        # competitors - dataclass defaults (0.0 diff, 1.0 ratio, rank 0, degree 0,
        # counts of 1 = self only) are the neutral "no competition" values.
        no_competition = asdict(CompetitionFeatures()) if self.ENABLE_COMPETITION_FEATURES else {}
        reject_rows: list[list[float]] = []

        for vehicle_id in reject_vehicle_ids:
            row_features: FeatureVector = {}
            row_features.update(state_features)
            row_features.update(self._vehicle_features(vehicles[vehicle_id], vehicles, current_time))
            # 2026-08-06: every reject row for this batch gets the SAME global_future_demand
            # (computed once above from the full requests pool, not per-vehicle) - so the model
            # can see the real batch-wide demand pressure behind each vehicle's reject option,
            # instead of always seeing an all-zero grid regardless of context.
            row_features.update(global_future_demand)
            row_features.update(no_candidate_location)
            row_features.update(no_competition)
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
        trip_costs: Iterable[TripCost] = getattr(
            trip_handler, "trip_costs", getattr(trip_handler, "trip_costs", [])
        )

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

    def _trip_features(
        self, trip: Union[Trip, SharedTrip], vehicle: Union[Vehicle, None]
    ) -> FeatureVector:
        """
        Trip-level features including normalized request count by vehicle max capacity.
        """
        # TODO fix this as the capacities must be considered separately for am and wc
        features = TripFeatures()

        if isinstance(trip, Trip):
            num_requests_in_trip = 1.0
        else:  # SharedTrip
            num_requests_in_trip = float(trip.cardinality)

        if vehicle is None:
            vehicle_max_capacity = 1.0
        else:
            full_am = float(getattr(vehicle, "full_am_capacity", vehicle.am_capacity))
            full_wc = float(getattr(vehicle, "full_wc_capacity", vehicle.wc_capacity))
            vehicle_max_capacity = max(full_am + full_wc, 1.0)

        features.t_num_requests_in_trip = num_requests_in_trip
        features.t_norm_num_requests_by_vehicle_max_capacity = (
            num_requests_in_trip / vehicle_max_capacity
        )
        return asdict(features)

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
        # features.tc_cost = float(trip_cost.cost)
        features.tc_sequence_len = len(plan.sequence)
        features.tc_num_trips = len(plan.trips)
        # if plan.added_cost >= 0:
        #     features.tc_added_cost = float(plan.added_cost)

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

        # TODO add remaining times in relation to the current_time for current dropoffs or remeinaing pickup times (basically potential for later detours)

        # print(trip_cost.trip_no, trip_cost.vehicle_id, features)
        return asdict(features)

    def _global_future_demand_features(self, requests: List[Request], current_time: float) -> FeatureVector:
        """
        Aggregates future request pickup demand onto a 7x7 geographic grid with interval-based
        temporal decay.

        2026-07-30: `requests` must be the FULL currently-known/pending request
        pool for this batch (e.g. build_from_components' own `requests` param,
        which is trip_handler.requests upstream) - NOT one trip candidate's own
        requests. This is a genuinely global, batch-level signal: identical for
        every trip_cost row and for the reject rows in the same
        build_from_components()/add_reject_action_entries() call, since it only
        depends on current_time and the batch's request pool. See
        _candidate_request_location_features for the (small, non-grid)
        candidate-specific counterpart this replaced within each row.

        The operating area (defined by min/max lat and lon) is divided into a 7x7 grid.
        For each request, the pickup origin is mapped to a grid cell and a decay
        score is accumulated based on pickup-time proximity:

          - Interval 1  pickup_time <= current_time + 1*BATCH_INTERVAL: score = 1.0
          - Interval 2  [current_time + BATCH_INTERVAL,  current_time + 2*BATCH_INTERVAL): score = 0.5
          - Interval 3  [current_time + 2*BATCH_INTERVAL, current_time + 3*BATCH_INTERVAL): score = 0.25
          - Interval 4  [current_time + 3*BATCH_INTERVAL, current_time + 4*BATCH_INTERVAL): score = 0.125
          - Beyond interval 4: ignored

        Requests whose earliest_pickup_time falls beyond the look-ahead horizon
        (4 * BATCH_INTERVAL) are ignored. Dropoff location and capacity are not used.

        Returns a 49-element FeatureVector keyed as ``gfd_grid_{row}_{col}`` where row 0 / col 0
        corresponds to the south-west corner of the operating area (min lat / min lon).

        Parameters
        ----------
        requests : List[Request]
            The full batch's currently-known/pending requests (not just one
            candidate's own requests). Requests already due
            (earliest_pickup_time <= current_time) are treated as immediate
            demand (interval 1).
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

            # Which 1-indexed interval does this pickup fall into?
            offset = pickup_time - current_time
            if offset <= 0:
                interval_index = 1
            else:
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
            f"gfd_grid_{row}_{col}": float(grid[row, col])
            for row in range(GRID_SIZE)
            for col in range(GRID_SIZE)
        }

    def _candidate_request_location_features(self, requests: List[Request], current_time: float) -> FeatureVector:
        """
        Compact candidate-specific counterpart to _global_future_demand_features:
        where this trip candidate's OWN request(s) are, and how urgent the most
        pressing one is. `requests` here is deliberately just the 1-2 (up to
        max_cardinality) requests already inside this candidate trip - the
        opposite scope of _global_future_demand_features's full batch pool.

        Mean pickup position keeps this cheap and cardinality-agnostic rather
        than padding per-request slots; urgency reuses the same interval-decay
        scale as the global grid (1.0 = due now, decaying to 0 beyond the
        4*BATCH_INTERVAL look-ahead horizon) so both features are on a
        comparable scale.
        """
        features = CandidateRequestFeatures()
        if not requests:
            return asdict(features)

        # Normalize to [0, 1] fractional position within the operating area,
        # same lat_range/lon_range basis _global_future_demand_features uses
        # to place a request into its grid cell - keeps both features on a
        # comparable [0, 1] scale and consistent with the 0.5 "center of map"
        # default (mirrors v_norm_lat_next_position's convention).
        lat_range = self.max_lat - self.min_lat
        lon_range = self.max_lon - self.min_lon
        lats = [request.origin.lat for request in requests]
        lons = [request.origin.lon for request in requests]
        mean_lat = sum(lats) / len(lats)
        mean_lon = sum(lons) / len(lons)
        features.cr_norm_lat = (mean_lat - self.min_lat) / lat_range if lat_range > 0 else 0.0
        features.cr_norm_lon = (mean_lon - self.min_lon) / lon_range if lon_range > 0 else 0.0

        best_decay = 0.0
        min_slack = 4.0
        for request in requests:
            offset = request.earliest_pickup_time - current_time
            if offset <= 0:
                interval_index = 1
            else:
                interval_index = int(offset // self.config.BATCH_INTERVAL) + 1
            if interval_index <= 4:  # beyond the look-ahead horizon, same cutoff as the global grid
                best_decay = max(best_decay, 0.5 ** (interval_index - 1))

            slack = (request.latest_pickup_time - current_time) / self.config.BATCH_INTERVAL
            min_slack = min(min_slack, slack)
        features.cr_urgency = best_decay
        features.cr_pickup_slack = min_slack

        return asdict(features)

    def _build_competition_lookup(
        self, trip_costs: List[TripCost]
    ) -> Tuple[Dict[int, List[TripCost]], Dict[int, List[TripCost]]]:
        """
        2026-08-09: groups trip_costs by vehicle_id and by request_id, once per
        batch - same "compute once, share across rows" pattern as
        _global_future_demand_features. Feeds _competition_features(). A
        candidate with 2 requests (SharedTrip) appears in both request groups.
        """
        by_vehicle: Dict[int, List[TripCost]] = {}
        by_request: Dict[int, List[TripCost]] = {}
        for tc in trip_costs:
            by_vehicle.setdefault(tc.vehicle_id, []).append(tc)
            for rid in tc.get_ordered_request_ids():
                by_request.setdefault(rid, []).append(tc)
        return by_vehicle, by_request

    def _competition_features(
        self,
        trip_cost: TripCost,
        by_vehicle: Dict[int, List[TripCost]],
        by_request: Dict[int, List[TripCost]],
    ) -> FeatureVector:
        """
        2026-08-09: how this candidate's cost compares to the alternatives that
        compete for the same vehicle and/or the same request(s) - see
        CompetitionFeatures. `by_vehicle`/`by_request` come from
        _build_competition_lookup(), computed once per batch and reused across
        all rows (not recomputed per candidate).
        """
        features = CompetitionFeatures()

        vehicle_alternatives = by_vehicle.get(trip_cost.vehicle_id, [trip_cost])
        features.cf_num_candidates_same_vehicle = len(vehicle_alternatives)
        best_vehicle_cost = min(tc.cost for tc in vehicle_alternatives)
        features.cf_cost_minus_best_for_vehicle = trip_cost.cost - best_vehicle_cost

        other_vehicle_costs = [tc.cost for tc in vehicle_alternatives if tc is not trip_cost]
        if other_vehicle_costs:
            mean_competing_cost = sum(other_vehicle_costs) / len(other_vehicle_costs)
            features.cf_diff_to_mean_competing_cost = trip_cost.cost - mean_competing_cost
            rank = 1 + sum(1 for c in other_vehicle_costs if c < trip_cost.cost)
            features.cf_norm_rank_for_vehicle = rank / len(vehicle_alternatives)
        # else: no other candidates for this vehicle - keep the dataclass defaults (0.0, 0.0)

        request_ids = trip_cost.get_ordered_request_ids()
        request_neighbors: set = set()
        best_request_costs: list = []
        for rid in request_ids:
            alternatives = by_request.get(rid, [trip_cost])
            request_neighbors.update(alternatives)
            best_request_costs.append(min(tc.cost for tc in alternatives))
        if best_request_costs:
            mean_best_request_cost = sum(best_request_costs) / len(best_request_costs)
            # 2026-08-09: floor the denominator instead of raising - added_cost can be
            # very small/near-zero in edge cases, and this is a soft ML feature, not a
            # hard constraint that should crash the pipeline over one candidate.
            features.cf_cost_ratio_best_for_request = trip_cost.cost / max(mean_best_request_cost, 1e-6)
        request_neighbors.discard(trip_cost)
        features.cf_num_candidates_same_request = len(request_neighbors) + 1

        conflict_neighbors = set(vehicle_alternatives) | request_neighbors
        conflict_neighbors.discard(trip_cost)
        features.cf_conflict_graph_degree = len(conflict_neighbors)

        return asdict(features)

    @staticmethod
    def _calc_geo_distance_meter(loc1: tuple[float, float], loc2: tuple[float, float]):
        """
        each location must be defined as a tuple with (lat, lon)
        
        # TODO if we use a travel_time_matrix, we should not need the geodesic distance but rather the travel_time_matrix values
        """
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