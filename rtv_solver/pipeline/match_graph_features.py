from __future__ import annotations

from typing import Sequence

import torch

from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.pipeline.feat_builder import FeatureBuilder
from rtv_solver.pipeline.match_solution_graph import MatchGraph
from rtv_solver.pipeline.request_graph_feature_builder import RequestGraphFeatureBuilder
from rtv_solver.structure.config import Config
from rtv_solver.structure.request import Request
from rtv_solver.structure.vehicle import Vehicle

# Order must match VehicleFeatures in feat_builder.py - we read this dict by
# name because _vehicle_features() returns a dict, not a fixed-order list.
VEHICLE_FEATURE_ORDER = [
    "v_norm_lat_next_position",
    "v_norm_lon_next_position",
    "v_norm_remaining_operating_period",
    "v_norm_vehicle_count_in_proximity",
    "v_avg_vehicle_distance",
    "v_norm_step_remaining_boarded_time",
    "v_norm_interval_remaining_boarded_time",
    "v_norm_remaining_am_cap",
    "v_norm_remaining_wc_cap",
    "v_am_cap",
    "v_wc_cap",
]


class MatchGraphFeatureBuilder:
    """
    Builds the two node-feature tensors (requests, vehicles) for the critic's
    MatchGraph (see match_solution_graph.py).

    Vehicle features are 11 values taken unchanged from the actor's
    FeatureBuilder._vehicle_features(). 2026-08-07: that method has a known
    distance-metric bug for Li&Lim (see figures_export/distance_metric_bug_slide.pptx)
    and two redundant features - we agreed to keep it as-is for V1 and only
    fix it once the critic works end-to-end.

    Request features are 15 values, mostly reused from RequestGraphFeatureBuilder
    (the request-pruner's feature code) plus a handful of new one-line features.
    Unlike the vehicle features, these are written fresh here, so they use the
    correct geographic-aware distance (Euclidean for Li&Lim grid coordinates,
    Haversine for NYC) from the start instead of inheriting the bug.

    Row order for both tensors must match MatchGraph.request_ids / vehicle_ids
    exactly, since the critic later indexes into the graph by position.
    """

    REQUEST_FEATURE_SIZE = 15
    VEHICLE_FEATURE_SIZE = 11

    def __init__(self, complete_payload: dict, config: Config, geographic: bool = False) -> None:
        self.config = config
        # 2026-08-07: reuses config.REQUEST_GRAPH_GEOGRAPHIC by convention (same
        # dataset the pruner runs on), passed in explicitly so this class does not
        # have to guess which config field to read.
        self.geographic = geographic
        self.distance_fn = (
            RequestGraphFeatureBuilder._haversine_distance_m
            if geographic
            else RequestGraphFeatureBuilder._euclidean_distance
        )

        (min_lat, max_lat), (min_lon, max_lon) = PayloadParser.get_request_operating_area_limits(complete_payload)
        self.min_lat, self.max_lat = min_lat, max_lat
        self.min_lon, self.max_lon = min_lon, max_lon
        # normalization denominators, computed with the correct distance metric
        # (not reused from FeatureBuilder, which always uses the buggy one)
        self.max_lat_distance = self.distance_fn(min_lat, max_lon, max_lat, max_lon)
        self.max_lon_distance = self.distance_fn(max_lat, min_lon, max_lat, max_lon)
        self.max_distance = self.distance_fn(min_lat, min_lon, max_lat, max_lon)

        r_start_time, r_end_time = PayloadParser.get_requests_time_interval(complete_payload)
        self.r_start_time = r_start_time
        self.total_operating_time = r_end_time - r_start_time

    def build(
        self,
        requests: Sequence[Request],
        vehicles: dict[int, Vehicle],
        active_requests: dict,
        match_graph: MatchGraph,
        current_time: float,
        actor_feature_builder: FeatureBuilder,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        requests must be the exact same list (same order) passed into
        MatchSolutionGraphBuilder.build() for this iteration - that is what
        fixes match_graph.request_ids' order.
        """
        if len(requests) != match_graph.num_requests:
            raise ValueError(
                f"requests ({len(requests)}) does not match match_graph.num_requests "
                f"({match_graph.num_requests}) - wrong list or wrong order passed in."
            )

        request_features = self._build_request_features(requests, active_requests, match_graph, current_time)
        vehicle_features = self._build_vehicle_features(match_graph, vehicles, current_time, actor_feature_builder)
        return request_features, vehicle_features

    def _build_request_features(
        self,
        requests: Sequence[Request],
        active_requests: dict,
        match_graph: MatchGraph,
        current_time: float,
    ) -> torch.Tensor:
        batch_interval = self.config.BATCH_INTERVAL
        rows: list[list[float]] = []

        for i, request in enumerate(requests):
            origin = request.origin
            destination = request.destination

            # 1-2: pickup location - lets the critic tell requests apart by where they are
            pickup_lat = self.distance_fn(self.min_lat, origin.lon, origin.lat, origin.lon) / self.max_lat_distance
            pickup_lon = self.distance_fn(origin.lat, self.min_lon, origin.lat, origin.lon) / self.max_lon_distance

            # 3-4: dropoff location - same idea, other end of the trip
            dropoff_lat = self.distance_fn(self.min_lat, destination.lon, destination.lat, destination.lon) / self.max_lat_distance
            dropoff_lon = self.distance_fn(destination.lat, self.min_lon, destination.lat, destination.lon) / self.max_lon_distance

            # 5: how flexible this request's own pickup window is (static property, does not change with current_time)
            pickup_window_length = (request.latest_pickup_time - request.earliest_pickup_time) / batch_interval

            # 6: where in the overall instance day this pickup falls (early vs. late in the schedule)
            earliest_pickup_norm = max(0.0, min(1.0, (request.earliest_pickup_time - self.r_start_time) / self.total_operating_time))

            # 7: how soon (from now) the pickup window actually opens
            # can go negative for already-active requests whose window opened earlier - not clipped yet, revisit if this causes issues
            time_to_earliest = (request.earliest_pickup_time - current_time) / batch_interval

            # 8: how much time is left before the hard pickup deadline - continuous version of the urgent flag below
            pickup_slack = (request.latest_pickup_time - current_time) / batch_interval

            # 9: direct trip length - cheap/expensive to serve, on its own
            trip_distance = self.distance_fn(origin.lat, origin.lon, destination.lat, destination.lon) / self.max_distance

            # 10-11: seat demand, kept separate since ambulatory and wheelchair capacity aren't interchangeable
            am_capacity = float(request.am_capacity)
            wc_capacity = float(request.wc_capacity)

            # 12: about to miss its pickup deadline - same rule RequestPruner already force-keeps requests by
            urgent_flag = 1.0 if (request.latest_pickup_time - current_time) <= self.config.STEP_SIZE else 0.0

            # 13: was this request already matched to a vehicle in an earlier iteration (re-deciding it may matter differently than a brand-new request)
            previously_assigned_flag = 1.0 if request.id in active_requests else 0.0

            # 14: did this request end up assigned in THIS iteration's actual solution - redundant with the graph edges, but easier for the encoder to read directly
            selected_flag = 1.0 if match_graph.is_assigned[i] else 0.0

            # 15: how much other demand is nearby in space and time - local pressure signal
            # NOTE: radius is tuned for Li&Lim grid units, not meaningful yet for NYC meters - same open issue as in RequestGraphFeatureBuilder, revisit later
            local_density = self._local_spatiotemporal_count(request, requests)

            rows.append([
                pickup_lat, pickup_lon, dropoff_lat, dropoff_lon,
                pickup_window_length, earliest_pickup_norm, time_to_earliest, pickup_slack,
                trip_distance, am_capacity, wc_capacity,
                urgent_flag, previously_assigned_flag, selected_flag, local_density,
            ])

        return torch.tensor(rows, dtype=torch.float32)

    def _local_spatiotemporal_count(
        self,
        request: Request,
        requests: Sequence[Request],
        radius: float = 10.0,
        window_seconds: float = 600.0,
    ) -> float:
        count = 0.0
        for other in requests:
            if other is request:
                continue
            spatial = self.distance_fn(request.origin.lat, request.origin.lon, other.origin.lat, other.origin.lon)
            temporal = abs(request.earliest_pickup_time - other.earliest_pickup_time)
            if spatial <= radius and temporal <= window_seconds:
                count += 1.0
        return count

    def _build_vehicle_features(
        self,
        match_graph: MatchGraph,
        vehicles: dict[int, Vehicle],
        current_time: float,
        actor_feature_builder: FeatureBuilder,
    ) -> torch.Tensor:
        rows: list[list[float]] = []
        for vehicle_id in match_graph.vehicle_ids:
            # reuses the actor's own vehicle-feature method as-is, known issues and all (see class docstring)
            feature_dict = actor_feature_builder._vehicle_features(vehicles[vehicle_id], vehicles, current_time)
            rows.append([float(feature_dict[name]) for name in VEHICLE_FEATURE_ORDER])
        return torch.tensor(rows, dtype=torch.float32)
