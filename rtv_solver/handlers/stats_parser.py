from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.handlers.network_handler import NetworkHandler
from rtv_solver.structure.node import Node
from rtv_solver.structure.vehicle_stop import VehicleStop

@dataclass
class StatsConfig:
    server_url: str = "http://127.0.0.1:5001/"
    travel_time_error_margin: int = 5
    pickup_dwell: int = 180
    dropoff_dwell: int = 60

@dataclass
class StopPair:
    pickup: Optional[dict] = None
    dropoff: Optional[dict] = None

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
    vmt: float = 0.0    # vehicle miles traveled (TODO seems like we only consider time in seconds)
    pmt: float = 0.0    # passenger miles traveled (TODO seconds)
    serviced: int = 0   # total serviced requests         

    wait_time: List[float] = field(default_factory=list) # list of wait times per vehicle
    detour: List[float] = field(default_factory=list) # list of detour times per vehicle

    vmt_over_pmt: float = 0.0       # higher is better
    average_wait_time: float = 0.0  # lower is better
    average_detour: float = 0.0     # lower is better

    def finalize(self) -> None:
        if self.pmt > 0:
            self.vmt_over_pmt = self.vmt / self.pmt
        if self.serviced > 0:
            self.average_wait_time = sum(self.wait_time) / self.serviced
            self.average_detour = sum(self.detour) / self.serviced

    def to_dict(self):
        return asdict(self)

class StatsParser:
    """
    Evaluate final feasibility + compute KPIs for a complete payload (incl. filled manifests) and unserved requests
    """
    def __init__(self, config: StatsConfig | None = None):
        self.config = config or StatsConfig()
        self.server_url = self.config.server_url

        # booking_id -> StopPair(pickup=..., dropoff=...)
        self.request_stops: Dict[int, StopPair] = {}

        self.stats = Stats()
        self.violations: List[Violation] = []

        self._network_initialized = False

    def evaluate(self, payload: dict, unserved_requests: list[dict]) -> Tuple[bool, Stats, List[Violation]]:
        """
        evaluate the final result in relation to the input data 

         Main outputs:
        - feasible: bool
        - stats: Stats
        - violations: list[Violation] (for debugging / analysis)

        TODO add evaluation / analysis of initial and unserved requests
        """
        depot = payload[PayloadParser.DEPOT]
        requests = payload[PayloadParser.REQUESTS]
        driver_runs = payload[PayloadParser.DRIVERS]

        self._init_network()

        self.request_stops.clear()
        self.stats = Stats()
        self.violations = []

        for driver_run in driver_runs:
            self._simulate_driver_run(depot, driver_run)

        self._compute_request_metrics()

        feasible = len(self.violations) == 0
        self.stats.finalize()
        return feasible, self.stats, self.violations

    def _init_network(self) -> None:
        if self._network_initialized:
            return
        NetworkHandler.init(True, self.server_url)
        self._network_initialized = True

    def _simulate_driver_run(self, depot: dict, driver_run: dict) -> None:
        # TODO check whether all the checks are equivalent to prior get_stats()
        state = driver_run[PayloadParser.DRIVER_STATE]
        manifest = driver_run[PayloadParser.DRIVER_MANIFEST]

        run_id = int(state[PayloadParser.DRIVER_STATE_RUN_ID])
        max_am_capacity = int(state[PayloadParser.DRIVER_STATE_AM_CAP])
        max_wc_capacity = int(state[PayloadParser.DRIVER_STATE_WC_CAP])

        current_node = self._node_from_depot(depot)
        current_time = state[PayloadParser.DRIVER_STATE_START_TIME]
        current_load_am = 0
        current_load_wc = 0

        for stop in manifest:
            booking_id = float(stop[PayloadParser.MANIFEST_BOOKING_ID])

            # ensure we have a container for this request
            if booking_id not in self.request_stops:
                self.request_stops[booking_id] = StopPair()

            # turn stop into VehicleStop and compare with those variables after the fact that handles manifest operations
            action = stop[PayloadParser.MANIFEST_ACTION]
            scheduled_time = stop[PayloadParser.MANIFEST_SCHED_TIME]
            tw_start = stop[PayloadParser.MANIFEST_TIME_WINDOW_START]
            tw_end = stop[PayloadParser.MANIFEST_TIME_WINDOW_END]

            next_node = self._node_from_stop(stop)

            # travel
            travel_time = NetworkHandler.travel_time(current_node, next_node)
            self.stats.vmt += travel_time
            arrival_time = current_time + travel_time

            # check "schedule impossible" (arrival after scheduled_time + margin)
            if arrival_time > scheduled_time + self.config.travel_time_error_margin:
                self._add_violation(
                    "Scheduled time is impossible given travel time", booking_id, run_id, stop, details={
                        "arrival_time": arrival_time,
                        "scheduled_time": scheduled_time,
                        "margin": self.config.travel_time_error_margin,
                        "lateness": arrival_time - scheduled_time,
                    })

            # allow waiting until scheduled_time (mirrors your original behavior)
            service_start = max(arrival_time, scheduled_time)

            # time window checks (TODO: old code checked scheduled_time, not actual service_start)
            if service_start < tw_start:
                self._add_violation(
                    "Served before time window start", booking_id, run_id, stop,
                    details={"service_start": service_start, "tw_start": tw_start},)
            if service_start > tw_end:
                self._add_violation(
                    "Served after time window end", booking_id, run_id, stop,
                    details={"service_start": service_start, "tw_end": tw_end})

            # apply pickup/dropoff logic
            am_delta = int(stop[PayloadParser.MANIFEST_AMBULATORY])
            wc_delta = int(stop[PayloadParser.MANIFEST_WHEELCHAIR])

            if action == VehicleStop.ACT_PICKUP:
                # update load
                current_load_am += am_delta
                current_load_wc += wc_delta

                # dwell time
                service_end = service_start + self.config.pickup_dwell

                # store pickup stop
                if self.request_stops[booking_id].pickup is not None:
                    self._add_violation("Pickup already exists", booking_id, run_id, stop)
                self.request_stops[booking_id].pickup = stop

            else:  # DROPOFF
                current_load_am -= am_delta
                current_load_wc -= wc_delta
                service_end = service_start + self.config.dropoff_dwell

                # dropoff ordering constraints
                if self.request_stops[booking_id].dropoff is not None:
                    self._add_violation("Dropoff already exists", booking_id, run_id, stop)

                if self.request_stops[booking_id].pickup is None:
                    self._add_violation("Dropoff before pickup", booking_id, run_id, stop)

                self.request_stops[booking_id].dropoff = stop

            # capacity checks
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

    def _compute_request_metrics(self) -> None:
        """
        After all runs were simulated, compute PMT, wait_time, detour.
        """
        for booking_id, pair in self.request_stops.items():
            if pair.pickup is None:
                self._add_violation("Request missing pickup", booking_id)
                continue
            if pair.dropoff is None:
                self._add_violation("Request not dropped off", booking_id)
                continue

            pickup = pair.pickup
            dropoff = pair.dropoff

            origin = self._node_from_stop(pickup)
            destination = self._node_from_stop(dropoff)

            direct_travel_time = NetworkHandler.travel_time(origin, destination)
            # TODO seems to be incorrect, check
            self.stats.pmt += direct_travel_time

            pickup_time = pickup[PayloadParser.MANIFEST_SCHED_TIME]
            pickup_tw_start = pickup[PayloadParser.MANIFEST_TIME_WINDOW_START]
            dropoff_time = dropoff[PayloadParser.MANIFEST_SCHED_TIME]

            self.stats.wait_time.append(pickup_time - pickup_tw_start)
            self.stats.detour.append((dropoff_time - pickup_time) - direct_travel_time)

            # count as serviced if it has a full pair
            self.stats.serviced += 1

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
        return Node(depot["pt"]["lat"], depot["pt"]["lon"])

    def _node_from_stop(self, stop: dict) -> Node:
        loc = stop[PayloadParser.MANIFEST_LOC]
        return Node(loc["lat"], loc["lon"])
