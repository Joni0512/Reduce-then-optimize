from rtv_solver.structure.payload import Payload
from rtv_solver.handlers.network_handler import NetworkHandler
from rtv_solver.structure.vehicle_stop import VehicleStop

import copy

class PayloadParser:
    """
    Handles the parsing of the initial payloads in both directions, importing and transforming data. 
    """
    # keys for payload dictionary that can be used globally
    DATE = "date"
    TIME_MATRIX = "time_matrix"
    CURRENT_TIME = "current_time"

    DEPOT = "depot"
    DEPOT_PT = 'pt'

    REQUESTS = "requests"
    REQ_BOOKING_ID = "booking_id" # translate all to Booking_ID
    REQ_PICKUP_PT = "pickup_pt"
    REQ_PICKUP_LAT = 'pickup_latitude'
    REQ_PICKUP_LON = 'pickup_longitude'
    REQ_PICKUP_NODE_ID = 'pickup_node_id'
    REQ_DROPOFF_PT = "dropoff_pt"
    REQ_DROPOFF_NODE_ID = 'dropoff_node_id'
    REQ_DROPOFF_LAT = 'dropoff_latitude'
    REQ_DROPOFF_LON = 'dropoff_longitude'
    REQ_PICKUP_WINDOW_START = 'pickup_time_window_start'
    REQ_PICKUP_WINDOW_END = 'pickup_time_window_end'
    REQ_DROPOFF_WINDOW_START = 'dropoff_time_window_start'
    REQ_DROPOFF_WINDOW_END = 'dropoff_time_window_end'
    REQ_AMBULATORY = 'am'
    REQ_WHEELCHAIR = 'wc'
    REQ_DWELL_PICKUP = 'dwell_pickup'
    REQ_DWELL_ALIGHT = 'dwell_alight'

    DRIVERS = "driver_runs"
    DRIVER_STATE = "state"
    DRIVER_STATE_RUN_ID = "run_id"
    DRIVER_STATE_START_TIME = "start_time"
    DRIVER_STATE_END_TIME = "end_time"
    DRIVER_STATE_AM_CAP = "am_capacity"
    DRIVER_STATE_WC_CAP = "wc_capacity"
    DRIVER_STATE_T_LOCS = "total_locations"
    DRIVER_STATE_LOC = "loc"
    DRIVER_STATE_DT_SEC = "location_dt_seconds"
    DRIVER_STATE_LOC_SERV = "locations_already_serviced"

    DRIVER_MANIFEST = "manifest"
    MANIFEST_RUN_ID = "run_id"
    MANIFEST_ORDER = "order"
    MANIFEST_ACTION = "action"
    MANIFEST_BOOKING_ID = "booking_id"
    MANIFEST_LOC = "loc"
    MANIFEST_AMBULATORY = "am"
    MANIFEST_WHEELCHAIR = "wc"
    MANIFEST_SCHED_TIME = "scheduled_time"
    MANIFEST_TIME_WINDOW_START = "time_window_start"
    MANIFEST_TIME_WINDOW_END = "time_window_end"

    STATS_ASSIGNMENT_DEVELOPMENT = "stats_assign_dev"   
    STATS_ASSIGNED = 'assigned_requests'
    STATS_UNSERVED = 'unserved_requests'
    STATS_BOARDED = 'boarded'
    STATS_DROPPED = 'dropped'

    @staticmethod
    def get_payload_object(payload, online=True) -> Payload:
        """
        Based on the inserted payload data, a new payload is created.
        Specific attention to requests as these are combined from new requests and still active or boarded requests stored in the vehicleManifests."""
        # initialize time-matrix if available
        travel_time_matrix = payload.get(PayloadParser.TIME_MATRIX)
        
        # get current time from all vehicles (prefer already progressed vehicles, fallback to earliest start time vehicles)
        # TODO fix current_time as it should be aligned with the offline Solver iterator, currently it just takes the most recent time of the vehicle? in addition the current_time is also never used
        SECONDS_IN_DAY = 24 * 3600
        driver_runs = payload[PayloadParser.DRIVERS]
        current_time = SECONDS_IN_DAY
        if online:
            start_times = []
            progressed_times = []
            for driver_run in driver_runs:
                state = driver_run[PayloadParser.DRIVER_STATE]
                start_time = state[PayloadParser.DRIVER_STATE_START_TIME]
                last_time = state[PayloadParser.DRIVER_STATE_DT_SEC]
                start_times.append(start_time)
                if last_time > start_time:
                    progressed_times.append(last_time)
            # FIXME current time in here is the time for the next action of the vehicle but not the current_time of the simulation state
            if progressed_times:
                current_time = min(progressed_times)
            else:
                current_time = min(start_times)
        
        # build list of active and boarded requests from vehicle manifests
        active_requests_data = {}
        boarded_requests_data = {}
        for driver_run in driver_runs:
            if PayloadParser.DRIVER_MANIFEST in driver_run:
                driver_state = driver_run[PayloadParser.DRIVER_STATE]
                driver_manifest = driver_run[PayloadParser.DRIVER_MANIFEST]
                # iterate over all manifest stops
                for index, stop in enumerate(driver_manifest):
                    stop_order = stop[PayloadParser.MANIFEST_ORDER]
                    booking_id = stop[PayloadParser.MANIFEST_BOOKING_ID]
                    if stop[PayloadParser.MANIFEST_ACTION] == VehicleStop.ACT_PICKUP:
                        request = PayloadParser._build_request_from_manifest_index(driver_manifest, index)
                        if stop_order <= driver_state[PayloadParser.DRIVER_STATE_LOC_SERV]:
                            # request is picked up as the vehicleState has already picked up the location
                            boarded_requests_data[booking_id] = request
                        else:
                            # request is assigned but not yet picked up
                            active_requests_data[booking_id] = request
                    else: # VehicleStop.ACT_DROPOFF
                        if stop_order <= driver_state[PayloadParser.DRIVER_STATE_LOC_SERV] and booking_id in boarded_requests_data:
                            # vehicle has already been dropped off and it has previously been boarded
                            del boarded_requests_data[booking_id]
        
        # combine requests from new payload and add active and boarded requests from manifests (see preparation above)
        raw_requests = payload.get(PayloadParser.REQUESTS, [])
        requests = [PayloadParser._build_request(request) for request in raw_requests]
        for req_id in active_requests_data: # request must be handled as they were already accepted
            requests.append(active_requests_data[req_id])
        active_requests_keys = list(active_requests_data.keys())
        for req_id in boarded_requests_data: # request must be handled as they are already on board
            requests.append(boarded_requests_data[req_id])
        boarded_requests_keys = list(boarded_requests_data.keys())

        # get depot location
        depot_data = payload[PayloadParser.DEPOT]
        depot_location = depot_data.get("loc") or depot_data.get("pt") # depends on payload input
        node_id = NetworkHandler.get_next_node_id(depot_location["lat"], depot_location["lon"])
        depot = NetworkHandler.get_node_from_manifest_location(depot_location, node_id)

        return Payload(travel_time_matrix, current_time, requests, boarded_requests_keys, active_requests_keys, driver_runs, depot)

    @staticmethod
    def get_requests_time_interval(payload) -> tuple[int, int]:
        """ iterate over all requests to get the earliest start time and latest end time """
        start_time = 24*3600
        end_time = 0

        for request in payload[PayloadParser.REQUESTS]:
            if request[PayloadParser.REQ_PICKUP_WINDOW_START] < start_time:
                start_time = request[PayloadParser.REQ_PICKUP_WINDOW_START]
            if request[PayloadParser.REQ_DROPOFF_WINDOW_END] > end_time:
                end_time = request[PayloadParser.REQ_DROPOFF_WINDOW_END]
        return start_time, end_time
    
    @staticmethod
    def _build_request_from_manifest_index(manifest, pick_up_index):
        stop = manifest[pick_up_index]
        booking_id = stop[PayloadParser.MANIFEST_BOOKING_ID]
        for drop_off_stop in manifest[pick_up_index+1:]:
            if drop_off_stop[PayloadParser.MANIFEST_BOOKING_ID] == booking_id:
                return PayloadParser._build_request_from_stops(stop, drop_off_stop)

    @staticmethod
    def _build_request_from_stops(pickup_stop, dropoff_stop):
        """builds request from two separate stops out of manifest"""
        request = {
            PayloadParser.REQ_BOOKING_ID:               pickup_stop[PayloadParser.MANIFEST_BOOKING_ID],
            PayloadParser.REQ_AMBULATORY:               pickup_stop[PayloadParser.MANIFEST_AMBULATORY],
            PayloadParser.REQ_WHEELCHAIR:               pickup_stop[PayloadParser.MANIFEST_WHEELCHAIR],
            PayloadParser.REQ_PICKUP_WINDOW_START:      pickup_stop[PayloadParser.MANIFEST_TIME_WINDOW_START],
            PayloadParser.REQ_PICKUP_WINDOW_END:        pickup_stop[PayloadParser.MANIFEST_TIME_WINDOW_END],
            PayloadParser.REQ_PICKUP_PT:                pickup_stop[PayloadParser.MANIFEST_LOC],
            PayloadParser.REQ_DROPOFF_WINDOW_START:     dropoff_stop[PayloadParser.MANIFEST_TIME_WINDOW_START],
            PayloadParser.REQ_DROPOFF_WINDOW_END:       dropoff_stop[PayloadParser.MANIFEST_TIME_WINDOW_END],
            PayloadParser.REQ_DROPOFF_PT:               dropoff_stop[PayloadParser.MANIFEST_LOC],
        }
        return request

    @staticmethod
    def _build_request(request_data):
        """no changes due to this method, but makes it easier to read"""
        request = {
            PayloadParser.REQ_BOOKING_ID:           request_data[PayloadParser.REQ_BOOKING_ID],
            PayloadParser.REQ_AMBULATORY:           request_data[PayloadParser.REQ_AMBULATORY],
            PayloadParser.REQ_WHEELCHAIR:           request_data[PayloadParser.REQ_WHEELCHAIR],
            PayloadParser.REQ_PICKUP_WINDOW_START:  request_data[PayloadParser.REQ_PICKUP_WINDOW_START], 
            PayloadParser.REQ_PICKUP_WINDOW_END:    request_data[PayloadParser.REQ_PICKUP_WINDOW_END],
            PayloadParser.REQ_PICKUP_PT:            request_data[PayloadParser.REQ_PICKUP_PT],
            PayloadParser.REQ_DROPOFF_WINDOW_START: request_data[PayloadParser.REQ_DROPOFF_WINDOW_START], 
            PayloadParser.REQ_DROPOFF_WINDOW_END:   request_data[PayloadParser.REQ_DROPOFF_WINDOW_END],
            PayloadParser.REQ_DROPOFF_PT:           request_data[PayloadParser.REQ_DROPOFF_PT],
        }
        return request

    @staticmethod
    def _is_canonical_structure(data: dict) -> bool:
        """
        Detects whether the JSON already matches the canonical structure in the 'wilson' format.
        """
        return (
            "driver_runs" in data
            and len(data["driver_runs"]) > 0
            and "state" in data["driver_runs"][0]
        )

    @staticmethod
    def normalize_to_canonical(data: dict) -> dict:
        """
        Converts the newer JSON structure from 'chattanooga' into the expected structure of 'wilson'. 
        For structural differences, see 'Documentation.md'. The changes are only additions and no prior information is lost.
        """
        if PayloadParser._is_canonical_structure(data):
            return data  # Nothing to do

        normalized = copy.deepcopy(data)

        depot_loc = normalized[PayloadParser.DEPOT][PayloadParser.DEPOT_PT]

        new_driver_runs = []
        for run in normalized[PayloadParser.DRIVERS]:
            state = {
                # copy old state
                PayloadParser.DRIVER_STATE_RUN_ID: run[PayloadParser.DRIVER_STATE_RUN_ID],
                PayloadParser.DRIVER_STATE_START_TIME: run[PayloadParser.DRIVER_STATE_START_TIME],
                PayloadParser.DRIVER_STATE_END_TIME: run[PayloadParser.DRIVER_STATE_END_TIME],
                PayloadParser.DRIVER_STATE_AM_CAP: run[PayloadParser.DRIVER_STATE_AM_CAP],
                PayloadParser.DRIVER_STATE_WC_CAP: run[PayloadParser.DRIVER_STATE_WC_CAP],
                # injected defaults
                PayloadParser.DRIVER_STATE_LOC_SERV: 0,
                PayloadParser.DRIVER_STATE_DT_SEC: 0,
                # initialize location at depot
                PayloadParser.DRIVER_STATE_LOC: {
                    "lat": depot_loc["lat"],
                    "lon": depot_loc["lon"],
                }
            }
            new_driver_runs.append({
                PayloadParser.DRIVER_STATE: state,
                PayloadParser.DRIVER_MANIFEST: []})

        normalized[PayloadParser.DRIVERS] = new_driver_runs

        return normalized

    # TODO
    # def update_requests(data):
    """
    As all requests have exactly 30 minutes of allowed and combined wait + detour time, which does not seem very realistic if the direct travel_time of the trip is below 5 minutes, the data should be updated before usage. The requests should be updated once and for all before data is used.
     
    This also offers the option to add randomized versions of the same requests to get more training data sets while keeping realism."""
        
