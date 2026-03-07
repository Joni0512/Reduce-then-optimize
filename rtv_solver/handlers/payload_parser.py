from rtv_solver.structure.payload import Payload
from rtv_solver.handlers.network_handler import NetworkHandler
from rtv_solver.structure.vehicle_stop import VehicleStop

import copy
import pickle
import json
from pathlib import Path
from typing import Any
import numpy as np

from rtv_solver.util.logger import BASIC_LOGGER, DATA_LOGGER
import logging

console_logger = logging.getLogger(BASIC_LOGGER)
data_logger = logging.getLogger(DATA_LOGGER)

class PayloadParser:
    """
    Handles the parsing of the initial payloads in both directions, importing and transforming data. 
    """
    # keys for payload dictionary that can be used globally
    DATE = "date"
    TIME_MATRIX = "travel_time_matrix"
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
    def load_input_data(input_file: Path | str):
        if isinstance(input_file, str):
            file_type = input_file.split(".")[-1]
        elif isinstance(input_file, Path):
            file_type = input_file.suffix[1:].lstrip(".")
        else:
            raise ValueError(f"Unsupported file type: {input_file}")
        console_logger.info(f"Loading input data from {input_file} with file type {file_type}")
        if file_type == "json":
            with open(input_file, 'r') as f:
                data = json.load(f)
        elif file_type == "pkl":
            with open(input_file, 'rb') as f:
                data = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        return PayloadParser.normalize_to_canonical(data)

    @staticmethod
    def get_payload_object(payload: dict[str: Any], online: bool=True) -> Payload:
        """
        Based on the inserted payload data, a new payload is created.
        Specific attention to requests as these are combined from new requests and still active or boarded requests stored in the vehicleManifests."""
        # initialize time-matrix if available
        travel_time_matrix = payload.get(PayloadParser.TIME_MATRIX)
        driver_runs = payload[PayloadParser.DRIVERS]
        
        # for OfflineSolver, get current_time from simulation
        # for OnlineSolver, get current_time from all vehicles (prefer already progressed vehicles, fallback to earliest start time vehicles)
        # NOTE payload.current_time has so far never been really used, we need it for the feature creation
        SECONDS_IN_DAY = 24 * 3600
        online_current_time = SECONDS_IN_DAY
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
                online_current_time = min(progressed_times)
            else:
                online_current_time = min(start_times)
        current_time = payload.get(PayloadParser.CURRENT_TIME, online_current_time)
        
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
    def get_request_count(payload) -> int:
        return (len(payload[PayloadParser.REQUESTS]))
    
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
    def get_request_positions(payload):
        """
        Compute pickups and dropoffs of all requests.

        The function supports both payloads where coordinates are stored directly on the
        request (via REQ_*_LAT / REQ_*_LON) and payloads where they are nested inside
        REQ_*_PT dictionaries with ``lat`` / ``lon`` keys. The depot location is also
        taken into account, if present.
        """
        pickup_lats: list[float] = []
        pickup_lons: list[float] = []
        dropoff_lats: list[float] = []
        dropoff_lons: list[float] = []

        # collect request coordinates
        for request in payload.get(PayloadParser.REQUESTS, []):
            # pickup
            pickup_pt = request.get(PayloadParser.REQ_PICKUP_PT)
            if pickup_pt is not None:
                p_lat = pickup_pt.get("lat")
                p_lon = pickup_pt.get("lon")
            else:
                p_lat = request.get(PayloadParser.REQ_PICKUP_LAT)
                p_lon = request.get(PayloadParser.REQ_PICKUP_LON)
            if p_lat is not None and p_lon is not None:
                pickup_lats.append(p_lat)
                pickup_lons.append(p_lon)

            # dropoff
            dropoff_pt = request.get(PayloadParser.REQ_DROPOFF_PT)
            if dropoff_pt is not None:
                d_lat = dropoff_pt.get("lat")
                d_lon = dropoff_pt.get("lon")
            else:
                d_lat = request.get(PayloadParser.REQ_DROPOFF_LAT)
                d_lon = request.get(PayloadParser.REQ_DROPOFF_LON)
            if d_lat is not None and d_lon is not None:
                dropoff_lats.append(d_lat)
                dropoff_lons.append(d_lon)

        # depot location (if available)
        depot_lat = depot_lon = None
        if PayloadParser.DEPOT in payload:
            depot_data = payload[PayloadParser.DEPOT]
            depot_loc = depot_data.get("loc") or depot_data.get(PayloadParser.DEPOT_PT)
            if depot_loc is not None:
                depot_lat = depot_loc.get("lat")
                depot_lon = depot_loc.get("lon")

        return pickup_lats, pickup_lons, dropoff_lats, dropoff_lons, depot_lat, depot_lon
    
    @staticmethod
    def get_request_operating_area_limits(payload):
        """
        Computes the main operating area of all requests based on latitude and longitude.

        Returns:
            ((min_lat, max_lat), (min_lon, max_lon))
        """
        pickup_lats, pickup_lons, dropoff_lats, dropoff_lons, depot_lat, depot_lon = PayloadParser.get_request_positions(payload)

        # compute bounds
        all_lats: list[float] = pickup_lats + dropoff_lats
        all_lons: list[float] = pickup_lons + dropoff_lons
        if depot_lat is not None and depot_lon is not None:
            all_lats.append(depot_lat)
            all_lons.append(depot_lon)

        if not all_lats or not all_lons:
            raise ValueError("No coordinate data found in payload to determine operating area.")

        min_lat = min(all_lats)
        max_lat = max(all_lats)
        min_lon = min(all_lons)
        max_lon = max(all_lons)
        return (min_lat, max_lat), (min_lon, max_lon)
    
    @staticmethod
    def get_vehicle_time_intervals(payload) -> list[tuple[int, int]]:
        """
        Returns the operating time interval for each vehicle (driver run) in the payload.

        Each tuple in the returned list is `(start_time, end_time)` in seconds and corresponds
        to one entry in the driver_runs, in the same order.
        """
        intervals: list[tuple[int, int]] = []

        for driver_run in payload[PayloadParser.DRIVERS]:
            state = driver_run[PayloadParser.DRIVER_STATE]
            start_time = state[PayloadParser.DRIVER_STATE_START_TIME]
            end_time = state[PayloadParser.DRIVER_STATE_END_TIME]
            intervals.append((start_time, end_time))

        return intervals
    
    @staticmethod
    def get_vehicle_count(payload) -> int:
        return (len(payload[PayloadParser.DRIVERS]))

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
            # # turn all request-booking ids into strings
            # for request in data[PayloadParser.REQUESTS]:
            #     request[PayloadParser.REQ_BOOKING_ID] = str(request[PayloadParser.REQ_BOOKING_ID])
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

    @staticmethod
    def build_test_case(
        data: dict[str, Any],
        max_requests: int = 12,
        max_vehicles: int = 1,
        save_file_path: str = None):
        """
        Build a test case with a given number of requests and vehicles.
        """
        requests = copy.deepcopy(data.get(PayloadParser.REQUESTS, []))

        REPEAT_COUNT = 6


        # remove the first 8 requests
        requests = requests[REPEAT_COUNT:]
        new_request = {
            "booking_id":"1",
            "am":1,
            "wc":0,
            "pickup_time_window_start":20000.0,
            "pickup_time_window_end":20100.0,
            "pickup_pt":{
                "lon":-77.930793762,
                "lat":35.780387878
            },
            "dropoff_time_window_start":20611.9,
            "dropoff_time_window_end":20711.9,
            "dropoff_pt":{
                "lon":-77.893867493,
                "lat":35.719944
            }
        }
        # NOTE current approach fixes the data instead of the code
        # insert new request 8 times but increment the booking_id by 1 each time, start with 8 and then increment backwards as we add them to the top of the list

        new_requests = []

        travel_time = 611.9
        pickup_begin = 20000 
        pickup_end = pickup_begin + 100 # instead of 60 for pickup dwell
        dropoff_begin = pickup_end + travel_time + REPEAT_COUNT * 100
        dropoff_end = dropoff_begin + 200 # instead of 180 for dropoff dwell
        for i in range(0, REPEAT_COUNT+1):
            c_request = copy.deepcopy(new_request)
            c_request["booking_id"] = np.float64(i)
            # times add dwell time to the pickup and dropoff time window 
            c_request["pickup_time_window_start"] = pickup_begin
            c_request["pickup_time_window_end"] = pickup_end
            c_request["dropoff_time_window_start"] = dropoff_begin
            c_request["dropoff_time_window_end"] = dropoff_end
            pickup_begin += 60 # make sure that not every order is possible and we reduce the combinatortics for this case
            pickup_end += 60
            dropoff_begin += 180
            dropoff_end += 180
            new_requests.append(c_request)
        # limit number of requests to max_requests
        new_requests.append(requests[25])
        new_requests.append(requests[26])
        new_requests.append(requests[27])

        updated_requests = []
        for i in range(0, REPEAT_COUNT+4):
            c_request = copy.deepcopy(new_requests[i])
            c_request["booking_id"] = np.float64(i)
            updated_requests.append(c_request)


        # limit number of vehicles to max_vehicles
        vehicles = data[PayloadParser.DRIVERS][:max_vehicles]
        depot = data[PayloadParser.DEPOT]

        # build a new payload set with certain rules of requests
        new_payload = {
            'requests': updated_requests,
            'driver_runs': vehicles,
            'depot': depot
        }

        # save file to json
        if save_file_path is not None:
            with open(save_file_path, 'wb') as f:
                pickle.dump(new_payload, f)
                # json.dump(new_payload, f) to read it, but our booking_ids must be np.float64 for the solver to work (thus JSON does not work)


if __name__ == "__main__":
    """
    analyse the payload (especially requests) in order to adapt it for custom experiments.

    quick script to check the changes
    """
    # TODO move this into a visual function
    import argparse
    import pandas as pd
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser(description='Arguments for the PayloadParser main script')
    parser.add_argument('--input_file', type=str, default='inputs/wilson/random_weekeday_2.pkl', help='Path to the input file')
    args = parser.parse_args()

    data = PayloadParser.load_input_data(args.input_file)

    PayloadParser.build_test_case(data, max_requests=12, max_vehicles=1, save_file_path='inputs/test_nc/test_12r_1v_repeat6_simple.pkl')

    
    