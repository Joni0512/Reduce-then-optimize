from __future__ import annotations

import argparse
import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from collections import defaultdict
from pathlib import Path

from rtv_solver.schema.payload_keys import PayloadKeys
from rtv_solver.handlers.network_handler import NetworkHandler
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.structure.node import Node
from rtv_solver.structure.vehicle_stop import VehicleStop

if TYPE_CHECKING:
    from rtv_solver.structure.config import Config

from rtv_solver.util.logger import BASIC_LOGGER, DATA_LOGGER
import logging

console_logger = logging.getLogger(BASIC_LOGGER)
data_logger = logging.getLogger(DATA_LOGGER)


@dataclass
class StopPair:
    pickup: Optional[dict] = None
    pickup_time: Optional[float] = None
    pickup_dwell: Optional[float] = None
    pickup_tw_start: Optional[float] = None
    pickup_tw_end: Optional[float] = None
    
    dropoff: Optional[dict] = None
    dropoff_time: Optional[float] = None
    dropoff_dwell: Optional[float] = None
    dropoff_tw_start: Optional[float] = None
    dropoff_tw_end: Optional[float] = None

    def get_pickup_service_end_time(self) -> float:
        return self.pickup_time + self.pickup_dwell

@dataclass
class Violation:
    message: str
    booking_id: Optional[int] = None
    run_id: Optional[int] = None
    stop: Optional[dict] = None
    details: Optional[dict] = None

@dataclass
class Stats:
    """
    TODO add more stats on the use of vehicles: total cars involved (just number of driver_runs), average capacity usage (complex?: how many vehicles are in the car for how long?)

    TODO add rolling horizon stats: average backlog (complex: vehicles to be picked up), 

    TODO alternative stats on RTV generation, cannot be handled by this class as this is not tracked in the dictionary at the moment
    """
    vmt: float = 0.0        # vehicle miles traveled (TODO seems like we only consider time in seconds)
    pmt: float = 0.0        # passenger miles traveled (TODO seems like we only consider time in seconds)
    serviced: int = 0       # total serviced requests, higher is better

    wait_time: List[float] = field(default_factory=list)                # list of wait times per vehicle before initial pickup
    detour: List[float] = field(default_factory=list)                   # list of detour times per vehicle after pickup compared to direct travel
    direct_travel_time: List[float] = field(default_factory=list)      # list of direct travel times between pickup and dropoff per request
    dropoff_goal_lateness: List[float] = field(default_factory=list)    # list of times after dropoff window has opened (no direct impact on stats but useful for debugging)

    vmt_over_pmt: float = 0.0           # higher is better
    vmt_over_pmt_woDepot: float = 0.0   # accounts for depot trips    
    average_wait_time: float = 0.0      # lower is better
    average_detour: float = 0.0         # lower is better
    average_dropoff_goal_lateness: float = 0.0

    total_requests: int = 0       # total requests that were part of original payload
    serviced_requests: list[float] = field(default_factory=list)  # list of serviced requests 
    
    depot_movements: int = 0        # must be equal to number of vehicles if config.return_depot = True
    depot_vmt: float = 0.0          # tracked to counterweigh vmt

    rebalancing_movements: int = 0  # can only be != 0 when config.rebalance = True
    rebalancing_vmt: float = 0.0    # lower is better

    total_time: float = 0.0
    # Diagnostics for better interpretability:
    # - per vehicle totals in meters
    # - leg-by-leg decomposition of timing between service starts
    per_vehicle_distance_m: Dict[int, float] = field(default_factory=dict)
    per_vehicle_legs: Dict[int, List[dict]] = field(default_factory=dict)
    total_distance_m: float = 0.0

    def finalize(self) -> None:
        if self.pmt > 0:
            self.vmt_over_pmt = self.vmt / self.pmt
            self.vmt_over_pmt_woDepot = (self.vmt - self.depot_vmt) / self.pmt
        if self.serviced > 0:
            self.average_wait_time = sum(self.wait_time) / self.serviced
            self.average_detour = sum(self.detour) / self.serviced
        if self.dropoff_goal_lateness:
            self.average_dropoff_goal_lateness = sum(self.dropoff_goal_lateness) / len(self.dropoff_goal_lateness)

    @staticmethod
    def format_top_level(data: dict) -> str:
        """ prepare data object to print it well for the logging """
        lines = ["{"]
        for i, (key, value) in enumerate(data.items()):
            if isinstance(value, list): # Convert list of floats to list of ints
                value = [int(round(v)) if isinstance(v, (float, int)) else v for v in value]
            val_str = json.dumps(value) # Convert child to a single line string
            comma = "," if i < len(data) - 1 else "" # Add a comma unless it's the last item
            lines.append(f'    "{key}": {val_str}{comma}')
        lines.append("}")
        return "\n".join(lines)
    
    def to_dict(self):
        return asdict(self)

    def format_vehicle_leg_report(self, max_legs_per_vehicle: int = 50) -> str:
        """
        Human-readable per-vehicle report to debug manifest timing:
        service_start_gap ~= previous_stop_dwell + travel_time + waiting_before_service
        """
        lines: List[str] = []
        lines.append("Vehicle Leg Report")
        lines.append("=" * 120)
        lines.append(
            f"Total distance: {self.total_distance_m:.1f} m)"
        )
        lines.append("Columns:")
        lines.append("  leg: row index of the leg for this vehicle")
        lines.append("  from_stop: previous serviced stop label")
        lines.append("  to_stop: current serviced stop label")
        lines.append("  dist_m: network travel distance between from_stop and to_stop (meters)")
        lines.append("  wait_s: waiting before service starts at to_stop (seconds) (meaning early arrival at stop)")
        lines.append("  service_dwell_s: service duration at to_stop (service_end - service_start)")
        lines.append("  goal_late_s: how late service_end at a dropoff is after dropoff time_window_end")
        lines.append("  gap_s: actual time between consecutive service starts")
        lines.append("  recon_gap_s: reconstructed gap = previous_stop_dwell + travel_time + wait_s")

        run_ids = sorted(self.per_vehicle_legs.keys())
        if not run_ids:
            lines.append("No vehicle legs found.")
            return "\n".join(lines)

        for run_id in run_ids:
            legs = self.per_vehicle_legs.get(run_id, [])
            total_m = self.per_vehicle_distance_m.get(run_id, 0.0)

            lines.append("-" * 120)
            lines.append(
                f"Vehicle {run_id} | total distance: {total_m:.1f} m | legs: {len(legs)}"
            )
            header = (
                f"{'leg':>4}  "
                f"{'from_stop':<18}  "
                f"{'to_stop':<18}  "
                f"{'dist_m':>10}  "
                f"{'wait_s':>8}  "
                f"{'service_dwell_s':>15}  "
                f"{'goal_late_s':>11}  "
                f"{'gap_s':>8}  "
                f"{'recon_gap_s':>12}"
            )
            lines.append(header)
            lines.append("-" * len(header))

            if max_legs_per_vehicle <= 0:
                legs_to_show = legs
            else:
                legs_to_show = legs[:max_legs_per_vehicle]

            for i, leg in enumerate(legs_to_show, start=1):
                lines.append(
                    f"{i:4d}  "
                    f"{str(leg['from_stop'])[:18]:<18}  "
                    f"{str(leg['to_stop'])[:18]:<18}  "
                    f"{leg['travel_distance_m']:10.1f}  "
                    f"{leg['waiting_before_service_s']:8.1f}  "
                    f"{leg['service_dwell_s']:15.1f}  "
                    f"{leg['dropoff_goal_late_s']:11.1f}  "
                    f"{leg['service_start_gap_s']:8.1f}  "
                    f"{leg['gap_reconstructed_s']:12.1f}"
                )

            if max_legs_per_vehicle > 0 and len(legs) > len(legs_to_show):
                lines.append(
                    f"... truncated {len(legs) - len(legs_to_show)} legs for vehicle {run_id}. "
                    "Use --max_legs_per_vehicle 0 to print all."
                )

        return "\n".join(lines)

    def to_printable_dict(self) -> dict:
        data = asdict(self)
        # Keep debug diagnostics out of the default stats print.
        for key in (
            "per_vehicle_distance_m",
            "per_vehicle_legs"
        ):
            data.pop(key, None)
        return data
    
    def __str__(self):
        return f"Stats:\n {self.format_top_level(self.to_printable_dict())}"

class StatsParser:
    """
    Evaluate final feasibility + compute KPIs for a complete payload (incl. filled manifests) and unserved requests

    Note on dwell handling:
    - `config.DWELL_PICKUP` and `config.DWELL_ALIGHT` are currently only fallback values used when a manifest stop does not carry explicit dwell/service-end values.
    - Long-term, manifests should always contain explicit dwell/service timings for each stop, so stats evaluation is decoupled from external config defaults.
    """
    def __init__(self, config: Config | None = None, payload: dict | None = None):
        self.config = config
        self._network_initialized = False
        # TODO clean this up, giving the payload either here or nowhere
        self._init_network(payload)

        # request booking_id -> StopPair(pickup=..., dropoff=...)
        self.request_stops: Dict[int, StopPair] = {}

        self.stats = Stats()
        self.violations: List[Violation] = []
        self.unserved: List[float] = []

        self.feature_payload = []

    def evaluate(self, payload: dict) -> Tuple[bool, Stats, List[Violation]]:
        """
        evaluate the final result in relation to the input data 

         Main outputs:
        - feasible: bool
        - stats: Stats
        - violations: list[Violation] (for debugging / analysis)

        TODO add evaluation / analysis of initial and unserved requests
        """
        depot = payload[PayloadKeys.DEPOT]
        requests = payload[PayloadKeys.REQUESTS]
        driver_runs = payload[PayloadKeys.DRIVERS]

        self.request_stops.clear()
        self.stats = Stats()
        self.violations = []
        
        for driver_run in driver_runs:
            self._simulate_driver_run(depot, driver_run)

        self._compute_request_metrics()

        feasible = len(self.violations) == 0
        self.stats.total_requests = len(requests)
        self.stats.finalize()

        self._print_per_request_stats(driver_runs)

        return feasible, self.stats, self.violations
    
    def add_total_time(self, total_time: float) -> None:
        self.stats.total_time = total_time

    def evaluate_development(self, payload: dict):
        requests = payload[PayloadKeys.REQUESTS]
        return self._compute_request_development(requests)

    def _init_network(self, payload: dict | None = None) -> None:
        if self._network_initialized:
            return
        server_url = getattr(self.config, "SERVER_URL", None) # enable run without config
        NetworkHandler.init_from_payload(payload=payload, server_url=server_url)
        self._network_initialized = True

    def _simulate_driver_run(self, depot: dict, driver_run: dict) -> None:
        """
        Simulate a single driver run and update the stats accordingly.
        """
        state = driver_run[PayloadKeys.DRIVER_STATE]
        manifest = driver_run[PayloadKeys.DRIVER_MANIFEST]

        run_id = int(state[PayloadKeys.DRIVER_STATE_RUN_ID])
        max_am_capacity = int(state[PayloadKeys.DRIVER_STATE_AM_CAP])
        max_wc_capacity = int(state[PayloadKeys.DRIVER_STATE_WC_CAP])

        current_node = self._node_from_depot(depot)
        # TODO FIXME this is the start time of its operations but might not be the actual trip start of the driver run
        current_time = state[PayloadKeys.DRIVER_STATE_START_TIME]
        current_load_am = 0
        current_load_wc = 0

        # Ensure output containers exist and are stable across runs.
        self.stats.per_vehicle_distance_m.setdefault(run_id, 0.0)
        self.stats.per_vehicle_legs.setdefault(run_id, [])

        # set values to create the right prev_state for stop legs
        previous_service_start = float(current_time)
        previous_service_dwell = 0.0
        previous_stop_label = "depot(start)"
        
        for stop in manifest:
            booking_id = stop[PayloadKeys.MANIFEST_BOOKING_ID]
            # ensure we have a container for this request
            if booking_id not in self.request_stops:
                self.request_stops[booking_id] = StopPair()

            # turn stop into VehicleStop and compare with those variables after the fact that handles manifest operations
            action = stop[PayloadKeys.MANIFEST_ACTION]
            scheduled_time = float(stop[PayloadKeys.MANIFEST_SCHED_TIME])
            tw_start = stop[PayloadKeys.MANIFEST_TIME_WINDOW_START]
            tw_end = stop[PayloadKeys.MANIFEST_TIME_WINDOW_END]

            next_node = self._node_from_stop(stop)

            # travel distance (per category) (NOTE time and distance can only be different if backend server is used)
            travel_time = NetworkHandler.travel_time(current_node, next_node)
            travel_distance_m = NetworkHandler.travel_distance(current_node, next_node)
            self.stats.vmt += travel_time
            self.stats.total_distance_m += travel_distance_m
            self.stats.per_vehicle_distance_m[run_id] += travel_distance_m
            
            if action == VehicleStop.ACT_DEPOT:
                self.stats.depot_vmt += travel_time
            if action == VehicleStop.ACT_REBALANCE:
                self.stats.rebalancing_vmt += travel_time
            
            # scheduled_time remains the canonical service start for backward compatibility
            service_start = float(stop.get(PayloadKeys.MANIFEST_SERVICE_START_TIME, scheduled_time))
            # On first leg (depot -> first stop): vehicle waits at depot until necessary to reach stop on time
            if previous_stop_label == "depot(start)":
                actual_departure = max(current_time, service_start - travel_time)
                computed_arrival_time = actual_departure + travel_time
            else:
                computed_arrival_time = current_time + travel_time
            # backward compatibile with old manifests
            manifest_arrival_time = float(stop.get(PayloadKeys.MANIFEST_ARRIVAL_TIME, computed_arrival_time))

            if action == VehicleStop.ACT_PICKUP:
                # NOTE FIXME this should be better, we need the dwell time from the trip and stops itself, not from the config; at least for LiLim it is part of the pickup_service_time and dropoff_service_time
                dwell_fallback = getattr(self.config, "DWELL_PICKUP", 180)
            elif action == VehicleStop.ACT_DROPOFF:
                dwell_fallback = getattr(self.config, "DWELL_ALIGHT", 60)
            else:
                dwell_fallback = 0
            # take dwell time stores in manifest for that stop if available, otherwise use fallback
            dwell_time = float(stop.get(PayloadKeys.MANIFEST_DWELL, dwell_fallback))
            service_end = float(stop.get(PayloadKeys.MANIFEST_SERVICE_END_TIME, service_start + dwell_time))

            if service_start < computed_arrival_time:
                console_logger.error("presumably the issue lies in the transition of node_ids which should never happen as then the travel_time between nodes changes and the simulation breaks. Current deep dive into the issue.")
                console_logger.error(f"Booking: {booking_id} action: {action} service_start < computed_arrival_time: {service_start} < {computed_arrival_time}")
            
            waiting_before_service = self._compute_waiting_before_service(service_start, computed_arrival_time)
            service_start_gap = service_start - previous_service_start
            service_dwell_s = self._compute_service_dwell(service_start, service_end)
            dropoff_goal_late_s = 0.0
            if action == VehicleStop.ACT_DROPOFF:
                dropoff_goal_late_s = self._compute_dropoff_tw_lateness(dropoff_service_start=service_start, dropoff_tw_start=tw_start)

            current_stop_label = f"{action}:{booking_id}"
            self.stats.per_vehicle_legs[run_id].append(
                {
                    "from_stop": previous_stop_label,
                    "to_stop": current_stop_label,
                    "travel_distance_m": travel_distance_m,
                    "travel_time_s": travel_time,
                    "waiting_before_service_s": waiting_before_service,
                    "service_dwell_s": service_dwell_s,
                    "dropoff_goal_late_s": dropoff_goal_late_s,
                    "service_start_gap_s": service_start_gap,
                    "gap_reconstructed_s": previous_service_dwell + travel_time + waiting_before_service,
                }
            )

            # check "schedule impossible" (arrival after scheduled_time + margin)
            travel_time_margin = getattr(self.config, "TRAVEL_TIME_MARGIN", 5)
            if computed_arrival_time > service_start + travel_time_margin:
                self._add_violation(
                    "Scheduled time is impossible given travel time", booking_id, run_id, stop, details={
                        "computed_arrival_time": computed_arrival_time,
                        "manifest_arrival_time": manifest_arrival_time,
                        "service_start_time": service_start,
                        "margin": travel_time_margin,
                        "lateness": computed_arrival_time - service_start,
                    })

            if service_end < service_start:
                self._add_violation(
                    "Service end before service start",
                    booking_id,
                    run_id,
                    stop,
                    details={
                        "service_start_time": service_start,
                        "service_end_time": service_end,
                    },
                )
                service_end = service_start

            # time window checks (NOTE: old code checked scheduled_time, not actual service_start)
            if service_start < tw_start:
                self._add_violation(
                    "Served before time window start", booking_id, run_id, stop,
                    details={"service_start": service_start, "tw_start": tw_start},)
            if service_start > tw_end:
                self._add_violation(
                    "Served after time window end", booking_id, run_id, stop,
                    details={"service_start": service_start, "tw_end": tw_end})

            # apply pickup/dropoff logic
            am_delta = int(stop[PayloadKeys.MANIFEST_AMBULATORY])
            wc_delta = int(stop[PayloadKeys.MANIFEST_WHEELCHAIR])

            if action == VehicleStop.ACT_PICKUP:
                # update load
                current_load_am += am_delta
                current_load_wc += wc_delta

                # store pickup stop
                if self.request_stops[booking_id].pickup is not None:
                    self._add_violation("Pickup already exists", booking_id, run_id, stop)
                self.request_stops[booking_id].pickup = stop
                self.request_stops[booking_id].pickup_time = service_start
                self.request_stops[booking_id].pickup_dwell = dwell_time
                self.request_stops[booking_id].pickup_tw_start = tw_start
                self.request_stops[booking_id].pickup_tw_end = tw_end
            elif action == VehicleStop.ACT_DROPOFF: 
                current_load_am -= am_delta
                current_load_wc -= wc_delta

                # dropoff ordering constraints
                if self.request_stops[booking_id].dropoff is not None:
                    self._add_violation("Dropoff already exists", booking_id, run_id, stop)
                if self.request_stops[booking_id].pickup is None:
                    self._add_violation("Dropoff before pickup", booking_id, run_id, stop)
                self.request_stops[booking_id].dropoff = stop
                self.request_stops[booking_id].dropoff_time = service_start
                self.request_stops[booking_id].dropoff_dwell = dwell_time
                self.request_stops[booking_id].dropoff_tw_start = tw_start
                self.request_stops[booking_id].dropoff_tw_end = tw_end
            elif action == VehicleStop.ACT_REBALANCE:
                self.stats.rebalancing_movements += 1
            elif action == VehicleStop.ACT_DEPOT:
                self.stats.depot_movements += 1
            else:
                self._add_violation("Unknown stop action", booking_id=booking_id, run_id=run_id)

            # capacity checks
            assert current_load_am >= 0, f"Ambulatory load cannot be negative (current_load_am={current_load_am})."
            assert current_load_wc >= 0, f"Wheelchair load cannot be negative (current_load_wc={current_load_wc})."
            if current_load_am > max_am_capacity:
                self._add_violation(
                    "Over capacity (ambulatory)",
                    booking_id=booking_id,
                    run_id=run_id,
                    stop=stop,
                    details={"am_load": current_load_am, "am_capacity": max_am_capacity},
                )
            if current_load_wc > max_wc_capacity:
                self._add_violation(
                    "Over capacity (wheelchair)",
                    booking_id=booking_id,
                    run_id=run_id,
                    stop=stop,
                    details={"wc_load": current_load_am, "wc_capacity": max_am_capacity},
                )

            # advance simulation state
            current_node = next_node
            current_time = service_end
            previous_service_start = service_start
            previous_service_dwell = max(service_end - service_start, 0.0)
            previous_stop_label = current_stop_label

    def _compute_request_metrics(self) -> None:
        """
        After all runs were simulated, compute PMT, wait_time, detour.
        """
        for booking_id, pair in self.request_stops.items():
            if pair.pickup is None:
                if booking_id >= 0: # negative value for depot requests; limits check to real passenger requests
                    self._add_violation("Request missing pickup", booking_id)
                continue
            if pair.dropoff is None:
                self._add_violation("Request not dropped off", booking_id)
                continue

            pickup_stop = pair.pickup
            dropoff_stop = pair.dropoff

            origin = self._node_from_stop(pickup_stop)
            destination = self._node_from_stop(dropoff_stop)

            direct_travel_time = NetworkHandler.travel_time(origin, destination)
            self.stats.pmt += direct_travel_time
            self.stats.direct_travel_time.append(direct_travel_time)

            pickup_time = pair.pickup_time
            pickup_tw_start = pair.pickup_tw_start
            pickup_tw_end = pair.pickup_tw_end
            dropoff_time = pair.dropoff_time
            dropoff_tw_start = pair.dropoff_tw_start
            dropoff_tw_end = pair.dropoff_tw_end

            self.stats.wait_time.append(pickup_time - pickup_tw_start)
            detour_time = self._compute_detour_time(pickup_service_end_time=pair.get_pickup_service_end_time(), dropoff_time=dropoff_time, direct_travel_time=direct_travel_time)
            self.stats.detour.append(detour_time)

            dropoff_goal_late_s = self._compute_dropoff_tw_lateness(
                dropoff_service_start=dropoff_time,
                dropoff_tw_start=dropoff_tw_start,
            )
            self.stats.dropoff_goal_lateness.append(dropoff_goal_late_s)

            # count as serviced if it has a full pair
            self.stats.serviced += 1
            self.stats.serviced_requests.append(booking_id)

    @staticmethod
    def _compute_detour_time(
        pickup_service_end_time: float,
        dropoff_time: float,
        direct_travel_time: float,
    ) -> float:
        """
        Compute detour time as extra in-vehicle time after pickup service end compared to direct travel.
        """
        actual_in_vehicle_time = dropoff_time - pickup_service_end_time
        detour_time = actual_in_vehicle_time - direct_travel_time
        assert detour_time >= -1e-6, f"Detour time cannot be negative (detour_time={detour_time}). Inconsistent timing inputs (pickup_service_end={pickup_service_end_time}, dropoff_time={dropoff_time}, direct_travel_time={direct_travel_time})."
        return max(detour_time, 0.0)

    @staticmethod
    def _compute_waiting_before_service(service_start: float, computed_arrival_time: float) -> float:
        """Compute waiting time before service starts at stop (meaning early arrival at stop)"""
        waiting = service_start - computed_arrival_time
        assert waiting >= -1e-6, f"Waiting before service cannot be negative (waiting={waiting}, service_start={service_start}, computed_arrival_time={computed_arrival_time})."
        return waiting

    @staticmethod
    def _compute_service_dwell(service_start: float, service_end: float) -> float:
        dwell = service_end - service_start
        assert dwell >= -1e-6, f"Service dwell cannot be negative (dwell={dwell})."
        return dwell

    @staticmethod
    def _compute_dropoff_tw_lateness(
        dropoff_service_start: float,
        dropoff_tw_start: float,
    ) -> float:
        lateness = dropoff_service_start - dropoff_tw_start
        assert lateness >= -1e-6, f"Lateness cannot be negative (lateness={lateness}, service_end={dropoff_service_start}, dropoff_tw_start={dropoff_tw_start})."
        return lateness

    def _compute_request_development(self, requests) -> None:
        """
        Computes the history of how request assignment developed, showing differences in assignment when keeping active_requests or not. This relates to the effect of the boolean flag --keep_active
        - boarded: requests that are currently inside a vehicle at that timestamp
        - delivered: requests that have been dropped off at or before that timestamp
        - assigned: requests that have been assigned in that timestamp to a vehicle
        - unserved: requests that have not been served in that timestamp

        TODO add changes based on vehicle, how often do requests change vehicles
        """
        assignment_history = {}
        
        output_dir = getattr(self.config, "OUTPUT_DIR", None)
        if output_dir is None:
            raise ValueError("evaluate_development requires config.OUTPUT_DIR to be set.")

        with open(Path(output_dir) / "assignment_data.jsonl", "r") as f:
            for line in f:
                entry = json.loads(line)

                sim_ts = int(entry["extra"]["timestamp"])
                assigned_status = entry["extra"]["status"][PayloadKeys.STATS_ASSIGNED]
                unserved_status = entry["extra"]["status"][PayloadKeys.STATS_UNSERVED] 

                assignment_history[sim_ts] = {}
                boarded: List[int] = []
                delivered: List[int] = []

                for req_id, sp in self.request_stops.items():
                    if req_id >= 0:
                        pu = sp.pickup_time
                        do = sp.dropoff_time
                        if pu <= sim_ts and sim_ts < do:
                            boarded.append(req_id)
                        elif sim_ts >= do:
                            delivered.append(req_id)
                    else: # artificial depot_request
                        continue
                
                assignment_history[sim_ts][PayloadKeys.STATS_BOARDED] = boarded
                assignment_history[sim_ts][PayloadKeys.STATS_DROPPED] = delivered # TODO only the dropped from the last step and not from all stops before

                per_vehicle = defaultdict(list)
                for req_id, veh_id in assigned_status.items():
                    per_vehicle[veh_id].append(req_id)

                assignment_history[sim_ts][PayloadKeys.STATS_ASSIGNED] = dict(per_vehicle)
                assignment_history[sim_ts][PayloadKeys.STATS_UNSERVED] = unserved_status

        return assignment_history
                
    def _add_violation(
        self,
        message: str,
        booking_id: Optional[int] = None,
        run_id: Optional[int] = None,
        stop: Optional[dict] = None,
        details: Optional[dict] = None,
    ) -> None:
        self.violations.append(
            Violation(
                message=message,
                booking_id=booking_id,
                run_id=run_id,
                stop=stop,
                details=details,
            )
        )

    def _node_from_depot(self, depot: dict) -> Node:
        return Node(depot["pt"]["lat"], depot["pt"]["lon"], depot.get("node_id", None))

    def _node_from_stop(self, stop: dict) -> Node:
        loc = stop["loc"]
        return Node(loc["lat"], loc["lon"], loc["node_id"])

    def _print_per_request_stats(self, driver_runs: list) -> None:
        header = f"{'id':>4}  {'wait_time':>10}  {'detour_time':>12}  {'direct_travel':>13}  {'dropoff_lateness':>16}  {'pickup_dwell':>12}  {'dropoff_dwell':>13}"
        print(header)
        print("-" * len(header))
        for i, bid in enumerate(self.stats.serviced_requests):
            bid_int = int(bid)
            pair = self.request_stops.get(bid_int)
            wait = self.stats.wait_time[i]
            detour = self.stats.detour[i]
            direct = self.stats.direct_travel_time[i] if i < len(self.stats.direct_travel_time) else 0.0
            pu_dwell = pair.pickup_dwell if pair and pair.pickup_dwell is not None else 0.0
            do_dwell = pair.dropoff_dwell if pair and pair.dropoff_dwell is not None else 0.0
            lateness = self.stats.dropoff_goal_lateness[i] if i < len(self.stats.dropoff_goal_lateness) else None
            lat_str = f"{lateness:.1f}" if lateness is not None else "N/A"
            print(f"{bid_int:4d}  {wait:10.1f}  {detour:12.1f}  {direct:13.1f}  {lat_str:>16}  {pu_dwell:12.1f}  {do_dwell:13.1f}")


def run_stats_from_result_dir(result_dir: Path | str) -> Tuple[bool, Stats, List[Violation]]:
    """
    Public helper to evaluate stats from a standard run directory that contains:
    - config.json
    - result_driver_runs.json
    """
    from rtv_solver.structure.config import Config

    run_dir = Path(result_dir).expanduser().resolve()
    config_file = run_dir / "config.json"
    result_manifest_file = run_dir / "result_driver_runs.json"

    with open(config_file, "r", encoding="utf-8") as f:
        config_json = json.load(f)
    config = Config.from_dict(config_json["config_dict"])
    config.OUTPUT_DIR = run_dir

    data = PayloadParser.load_input_data(result_manifest_file)
    stats_parser = StatsParser(config=config, payload=data)
    return stats_parser.evaluate(data)


def run_stats_from_manifest_file(
    manifest_file: Path | str,
    config: Config | None = None,
) -> Tuple[bool, Stats, List[Violation]]:
    """
    Public helper to evaluate stats directly from a canonical payload/manifest file.
    This path does not require a config.json file.
    """
    data = PayloadParser.load_input_data(manifest_file)
    stats_parser = StatsParser(config=config, payload=data)
    return stats_parser.evaluate(data)


def print_stats_report(
    feasible: bool,
    stats: Stats,
    violations: List[Violation],
    print_vehicle_legs: bool = False,
    max_legs_per_vehicle: int = 50,
) -> None:
    """Public helper to print human-readable stats output."""
    print(stats)
    if print_vehicle_legs:
        print(stats.format_vehicle_leg_report(max_legs_per_vehicle=max_legs_per_vehicle))
    print(f"feasible={feasible}, violations={len(violations)}")


if __name__ == "__main__":
    """
    get results from some of the LiLim datasets and calculate stats
    `python -m rtv_solver.handlers.stats_parser --input_file solutions/li_lim/manifests/lc101.json`

    get results from a run directory
    `python -m rtv_solver.handlers.stats_parser --result_dir outputs/debug/run_20260313_183009_bec60b --print_vehicle_legs`
    """
    parser = argparse.ArgumentParser(description="Arguments for the StatsParser main script")
    parser.add_argument(
        "--result_dir",
        type=str,
        default="outputs/debug/run_20260313_183009_bec60b",
        help="Path to run results directory containing config.json and result_driver_runs.json",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to payload/manifest file (e.g., LiLim manifest JSON) that can be evaluated without config.json.",
    )
    parser.add_argument(
        "--max_legs_per_vehicle",
        type=int,
        default=50,
        help="Max printed leg rows per vehicle in debug report (0 = print all).",
    )
    parser.add_argument(
        "--print_vehicle_legs",
        action="store_true",
        help="Print the per-vehicle leg breakdown report.",
    )
    args = parser.parse_args()

    if args.input_file is not None:
        feasible, stats, violations = run_stats_from_manifest_file(args.input_file)
        # For direct manifest debugging (e.g. LiLim optimal solutions), print legs by default.
        should_print_vehicle_legs = True
    else:
        feasible, stats, violations = run_stats_from_result_dir(args.result_dir)
        should_print_vehicle_legs = args.print_vehicle_legs

    print_stats_report(
        feasible=feasible,
        stats=stats,
        violations=violations,
        print_vehicle_legs=should_print_vehicle_legs,
        max_legs_per_vehicle=args.max_legs_per_vehicle,
    )

