from rtv_solver.structure.payload import Payload
from rtv_solver.handlers.network_handler import NetworkHandler
from rtv_solver.structure.vehicle_stop import VehicleStop

class PayloadParser:
    # keys for payload dictionary that can be used globally
    TIME_MATRIX = "time_matrix"

    DEPOT = "depot"

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

    @staticmethod
    def get_payload_object(payload, online=True) -> Payload:
        # initialize time-matrix if available
        travel_time_matrix = payload.get(PayloadParser.TIME_MATRIX)
        
        # get current time from all vehicles (prefer already progressed vehicles, fallback to earliest start time vehicles)
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
            if progressed_times:
                current_time = min(progressed_times)
            else:
                current_time = min(start_times)
        
        # build lists of active and boarded requests from vehicle manifests
        active_requests_data = {}
        boarded_requests_data = {}
        for driver_run in driver_runs:
            added_active_requests = []
            if PayloadParser.DRIVER_MANIFEST in driver_run:
                driver_state = driver_run[PayloadParser.DRIVER_STATE]
                driver_manifest = driver_run[PayloadParser.DRIVER_MANIFEST]
                # iterate over all manifest stops
                for index, stop in enumerate(driver_manifest):
                    stop_order = stop[PayloadParser.MANIFEST_ORDER]
                    booking_id = stop[PayloadParser.MANIFEST_BOOKING_ID]
                    if stop[PayloadParser.MANIFEST_ACTION] == VehicleStop.PICKUP:
                        request = PayloadParser.build_request_from_manifest_index(driver_manifest, index)
                        if stop_order <= driver_state[PayloadParser.DRIVER_STATE_LOC_SERV]:
                            boarded_requests_data[booking_id] = request
                        else:
                            added_active_requests.append(booking_id)
                            active_requests_data[booking_id] = request
                    else: # StopType.DROPOFF.value
                        if stop_order <= driver_state[PayloadParser.DRIVER_STATE_LOC_SERV] and booking_id in boarded_requests_data:
                            del boarded_requests_data[booking_id]
        
        # get requests from data and add active and boarded requests from manifests (see above)
        raw_requests = payload.get(PayloadParser.REQUESTS, [])
        requests = [PayloadParser.build_request(request) for request in raw_requests]
        for req_id in active_requests_data:
            requests.append(active_requests_data[req_id])
        active_requests = list(active_requests_data.keys())
        for req_id in boarded_requests_data:
            requests.append(boarded_requests_data[req_id])
        boarded_requests = list(boarded_requests_data.keys())

        # get depot location
        depot_data = payload[PayloadParser.DEPOT]
        depot_location = depot_data.get("loc") or depot_data.get("pt") # depends on payload input
        node_id = NetworkHandler.get_next_node_id(depot_location["lat"], depot_location["lon"])
        depot = NetworkHandler.manifest_location(depot_location, node_id)

        return Payload(travel_time_matrix, current_time, requests, boarded_requests, active_requests, driver_runs, depot)

    @staticmethod
    def get_requests_time_interval(payload) -> tuple[int, int]:
        """ iterate over all requests to get the earliest start time and latest end time """
        start_time = 24*3600
        end_time = 0

        for request in payload["requests"]:
            if request["pickup_time_window_start"] < start_time:
                start_time = request["pickup_time_window_start"]
            if request["dropoff_time_window_end"] > end_time:
                end_time = request["dropoff_time_window_end"]
        return start_time, end_time
    
    # TODO comment the code parts below in order to explain their purpose
    @staticmethod
    def build_request_from_manifest_index(manifest, pick_up_index):
        stop = manifest[pick_up_index]
        booking_id = stop[PayloadParser.MANIFEST_BOOKING_ID]
        for drop_off_stop in manifest[pick_up_index+1:]:
            if drop_off_stop[PayloadParser.MANIFEST_BOOKING_ID] == booking_id:
                return PayloadParser.build_request_from_stops(stop, drop_off_stop)
  
    @staticmethod  
    def build_request_from_manifest_dropoff(manifest, dropoff_stop):
        # seems to be deprecated as it is not used anywhere
        booking_id = dropoff_stop[PayloadParser.MANIFEST_BOOKING_ID]
        for pickup_stop in manifest:
            if pickup_stop[PayloadParser.MANIFEST_BOOKING_ID] == booking_id and pickup_stop[PayloadParser.MANIFEST_ACTION] == VehicleStop.PICKUP:
                return PayloadParser.build_request_from_stops(pickup_stop, dropoff_stop)

    @staticmethod
    def build_request_from_stops(pickup_stop, dropoff_stop):
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
    def build_request(request_data):
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
