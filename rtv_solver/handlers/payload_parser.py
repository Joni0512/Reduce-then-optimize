from rtv_solver.structure.payload import Payload
from rtv_solver.handlers.network_handler import NetworkHandler
from rtv_solver.structure.vehicle_stop import VehicleStop

import copy
import pickle
import json
from pathlib import Path
from typing import Any
import numpy as np

from rtv_solver.schema.payload_keys import PayloadKeys
from rtv_solver.parser.li_lim_parser import LiLimParser
from rtv_solver.parser.sartori_parser import SartoriParser

from rtv_solver.util.logger import BASIC_LOGGER, DATA_LOGGER
import logging

console_logger = logging.getLogger(BASIC_LOGGER)
data_logger = logging.getLogger(DATA_LOGGER)

class PayloadParser:
    """
    Handles the parsing of the initial payloads in both directions, importing and transforming data. 
    """

    @staticmethod
    def load_input_data(input_file: Path | str):
        """
        Loads the input data from the given file, checks which schema format conversion needs to be applied and returns a payload object that all solvers can interact with. 

        It is especially important to make sure that the time matrix is available for the solver to use in order to run the solver on the cluster without the backend server.
        # TODO time_matrix availability is currently not checked for the wilson format, this should be added.
        """
        if not isinstance(input_file, (str, Path)):
            raise ValueError(f"Unsupported file type: {input_file}")
        input_path = Path(input_file)
        file_type = input_path.suffix[1:].lstrip(".").lower()
        console_logger.info(f"Loading input data from {input_path} with file type {file_type}")
        if file_type == "json":
            with open(input_path, "r") as f:
                data = json.load(f)
        elif file_type == "pkl":
            with open(input_path, "rb") as f:
                data = pickle.load(f)
        elif file_type == "txt":
            txt_format = PayloadParser._detect_txt_format(input_path)
            if txt_format == "sartori":
                data = SartoriParser.parse_file(str(input_path))
            elif txt_format == "li_lim":
                data = LiLimParser.parse_file(str(input_path))
            else:
                raise ValueError(
                    f"Unsupported txt benchmark format in file: {input_path}. "
                    "Expected Sartori or Li-Lim instance format."
                )
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        return PayloadParser.normalize_to_canonical(data)

    @staticmethod
    def _detect_txt_format(input_path: Path) -> str | None:
        """
        Detect benchmark txt format based on path tokens and file signatures.
        """
        input_path_lc = str(input_path).lower()
        if "sartori" in input_path_lc:
            return "sartori"
        if "li_lim" in input_path_lc or "lilim" in input_path_lc:
            return "li_lim"

        with open(input_path, "r", encoding="utf-8") as file:
            non_empty_lines = [line.strip() for line in file if line.strip()]

        if not non_empty_lines:
            return None

        first_30_lines = non_empty_lines[:30]
        has_sartori_header = any(
            line.startswith(("NAME:", "LOCATION:", "TYPE:", "SIZE:", "CAPACITY:", "ROUTE-TIME:"))
            for line in first_30_lines
        )
        has_nodes_section = any(line == "NODES" for line in first_30_lines)
        has_edges_section = any(line == "EDGES" for line in non_empty_lines[:200])
        if has_sartori_header and (has_nodes_section or has_edges_section):
            return "sartori"

        first_line_parts = non_empty_lines[0].replace("\t", " ").split()
        if len(first_line_parts) == 3:
            try:
                int(first_line_parts[0])
                int(first_line_parts[1])
                float(first_line_parts[2])
                return "li_lim"
            except ValueError:
                pass

        for line in non_empty_lines[1:6]:
            parts = line.replace("\t", " ").split()
            if len(parts) >= 9:
                try:
                    int(parts[0])
                    float(parts[1])
                    float(parts[2])
                    int(parts[3])
                    int(parts[4])
                    int(parts[5])
                    int(parts[6])
                    int(parts[7])
                    int(parts[8])
                    return "li_lim"
                except ValueError:
                    continue

        return None

    @staticmethod
    def get_payload_object(payload: dict[str: Any], online: bool=True) -> Payload:
        """
        Based on the inserted payload data, a new payload is created.
        Specific attention to requests as these are combined from new requests and still active or boarded requests stored in the vehicleManifests."""
        # initialize time-matrix if available
        travel_time_matrix = payload.get(PayloadKeys.TIME_MATRIX)
        driver_runs = payload[PayloadKeys.DRIVERS]
        
        # for OfflineSolver, get current_time from simulation
        # for OnlineSolver, get current_time from all vehicles (prefer already progressed vehicles, fallback to earliest start time vehicles)
        # NOTE payload.current_time has so far never been really used, we need it for the feature creation
        SECONDS_IN_DAY = 24 * 3600
        online_current_time = SECONDS_IN_DAY
        if online:
            start_times = []
            progressed_times = []
            for driver_run in driver_runs:
                state = driver_run[PayloadKeys.DRIVER_STATE]
                start_time = state[PayloadKeys.DRIVER_STATE_START_TIME]
                last_time = state[PayloadKeys.DRIVER_STATE_DT_SEC]
                start_times.append(start_time)
                if last_time > start_time:
                    progressed_times.append(last_time)
            # FIXME current time in here is the time for the next action of the vehicle but not the current_time of the simulation state
            if progressed_times:
                online_current_time = min(progressed_times)
            else:
                online_current_time = min(start_times)
        current_time = payload.get(PayloadKeys.CURRENT_TIME, online_current_time)
        
        # build list of active and boarded requests from vehicle manifests
        active_requests_data = {}
        boarded_requests_data = {}
        for driver_run in driver_runs:
            if PayloadKeys.DRIVER_MANIFEST in driver_run:
                driver_state = driver_run[PayloadKeys.DRIVER_STATE]
                driver_manifest = driver_run[PayloadKeys.DRIVER_MANIFEST]
                # iterate over all manifest stops
                for index, stop in enumerate(driver_manifest):
                    stop_order = stop[PayloadKeys.MANIFEST_ORDER]
                    booking_id = stop[PayloadKeys.MANIFEST_BOOKING_ID]
                    if stop[PayloadKeys.MANIFEST_ACTION] == VehicleStop.ACT_PICKUP:
                        request = PayloadParser._build_request_from_manifest_index(driver_manifest, index)
                        if stop_order <= driver_state[PayloadKeys.DRIVER_STATE_LOC_SERV]:
                            # request is picked up as the vehicleState has already picked up the location
                            boarded_requests_data[booking_id] = request
                        else:
                            # request is assigned but not yet picked up
                            active_requests_data[booking_id] = request
                    else: # VehicleStop.ACT_DROPOFF
                        if stop_order <= driver_state[PayloadKeys.DRIVER_STATE_LOC_SERV] and booking_id in boarded_requests_data:
                            # vehicle has already been dropped off and it has previously been boarded
                            del boarded_requests_data[booking_id]
        
        # combine requests from new payload and add active and boarded requests from manifests (see preparation above)
        raw_requests = payload.get(PayloadKeys.REQUESTS, [])
        requests = [PayloadParser._build_request(request) for request in raw_requests]
        for req_id in active_requests_data: # request must be handled as they were already accepted
            requests.append(active_requests_data[req_id])
        active_requests_keys = list(active_requests_data.keys())
        for req_id in boarded_requests_data: # request must be handled as they are already on board
            requests.append(boarded_requests_data[req_id])
        boarded_requests_keys = list(boarded_requests_data.keys())

        # get depot location
        depot_data = payload[PayloadKeys.DEPOT]
        depot_location = depot_data.get("loc") or depot_data.get("pt") # depends on payload input
        node_id = NetworkHandler.get_next_node_id(depot_location["lat"], depot_location["lon"])
        depot = NetworkHandler.get_node_from_manifest_location(depot_location, node_id)

        return Payload(travel_time_matrix, current_time, requests, boarded_requests_keys, active_requests_keys, driver_runs, depot)

    @staticmethod
    def get_request_count(payload) -> int:
        return (len(payload[PayloadKeys.REQUESTS]))
    
    @staticmethod
    def get_requests_time_interval(payload) -> tuple[int, int]:
        """ iterate over all requests to get the earliest start time and latest end time """
        start_time = 24*3600
        end_time = 0

        for request in payload[PayloadKeys.REQUESTS]:
            if request[PayloadKeys.REQ_PICKUP_WINDOW_START] < start_time:
                start_time = request[PayloadKeys.REQ_PICKUP_WINDOW_START]
            if request[PayloadKeys.REQ_DROPOFF_WINDOW_END] > end_time:
                end_time = request[PayloadKeys.REQ_DROPOFF_WINDOW_END]
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
        for request in payload.get(PayloadKeys.REQUESTS, []):
            # pickup
            pickup_pt = request.get(PayloadKeys.REQ_PICKUP_PT)
            if pickup_pt is not None:
                p_lat = pickup_pt.get("lat")
                p_lon = pickup_pt.get("lon")
            else:
                p_lat = request.get(PayloadKeys.REQ_PICKUP_LAT)
                p_lon = request.get(PayloadKeys.REQ_PICKUP_LON)
            if p_lat is not None and p_lon is not None:
                pickup_lats.append(p_lat)
                pickup_lons.append(p_lon)

            # dropoff
            dropoff_pt = request.get(PayloadKeys.REQ_DROPOFF_PT)
            if dropoff_pt is not None:
                d_lat = dropoff_pt.get("lat")
                d_lon = dropoff_pt.get("lon")
            else:
                d_lat = request.get(PayloadKeys.REQ_DROPOFF_LAT)
                d_lon = request.get(PayloadKeys.REQ_DROPOFF_LON)
            if d_lat is not None and d_lon is not None:
                dropoff_lats.append(d_lat)
                dropoff_lons.append(d_lon)

        # depot location (if available)
        depot_lat = depot_lon = None
        if PayloadKeys.DEPOT in payload:
            depot_data = payload[PayloadKeys.DEPOT]
            depot_loc = depot_data.get("loc") or depot_data.get(PayloadKeys.DEPOT_PT)
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

        for driver_run in payload[PayloadKeys.DRIVERS]:
            state = driver_run[PayloadKeys.DRIVER_STATE]
            start_time = state[PayloadKeys.DRIVER_STATE_START_TIME]
            end_time = state[PayloadKeys.DRIVER_STATE_END_TIME]
            intervals.append((start_time, end_time))

        return intervals
    
    @staticmethod
    def get_vehicle_count(payload) -> int:
        return (len(payload[PayloadKeys.DRIVERS]))

    @staticmethod
    def _build_request_from_manifest_index(manifest, pick_up_index):
        stop = manifest[pick_up_index]
        booking_id = stop[PayloadKeys.MANIFEST_BOOKING_ID]
        for drop_off_stop in manifest[pick_up_index+1:]:
            if drop_off_stop[PayloadKeys.MANIFEST_BOOKING_ID] == booking_id:
                return PayloadParser._build_request_from_stops(stop, drop_off_stop)

    @staticmethod
    def _build_request_from_stops(pickup_stop, dropoff_stop):
        """builds request from two separate stops out of manifest"""
        request = {
            PayloadKeys.REQ_BOOKING_ID:               pickup_stop[PayloadKeys.MANIFEST_BOOKING_ID],
            PayloadKeys.REQ_AMBULATORY:               pickup_stop[PayloadKeys.MANIFEST_AMBULATORY],
            PayloadKeys.REQ_WHEELCHAIR:               pickup_stop[PayloadKeys.MANIFEST_WHEELCHAIR],
            PayloadKeys.REQ_PICKUP_WINDOW_START:      pickup_stop[PayloadKeys.MANIFEST_TIME_WINDOW_START],
            PayloadKeys.REQ_PICKUP_WINDOW_END:        pickup_stop[PayloadKeys.MANIFEST_TIME_WINDOW_END],
            PayloadKeys.REQ_PICKUP_PT:                pickup_stop[PayloadKeys.MANIFEST_LOC],
            PayloadKeys.REQ_DROPOFF_WINDOW_START:     dropoff_stop[PayloadKeys.MANIFEST_TIME_WINDOW_START],
            PayloadKeys.REQ_DROPOFF_WINDOW_END:       dropoff_stop[PayloadKeys.MANIFEST_TIME_WINDOW_END],
            PayloadKeys.REQ_DROPOFF_PT:               dropoff_stop[PayloadKeys.MANIFEST_LOC],
        }
        return request

    @staticmethod
    def _build_request(request_data):
        """no changes due to this method, but makes it easier to read"""
        request = {
            PayloadKeys.REQ_BOOKING_ID:           request_data[PayloadKeys.REQ_BOOKING_ID],
            PayloadKeys.REQ_AMBULATORY:           request_data[PayloadKeys.REQ_AMBULATORY],
            PayloadKeys.REQ_WHEELCHAIR:           request_data[PayloadKeys.REQ_WHEELCHAIR],
            PayloadKeys.REQ_PICKUP_WINDOW_START:  request_data[PayloadKeys.REQ_PICKUP_WINDOW_START], 
            PayloadKeys.REQ_PICKUP_WINDOW_END:    request_data[PayloadKeys.REQ_PICKUP_WINDOW_END],
            PayloadKeys.REQ_PICKUP_PT:            request_data[PayloadKeys.REQ_PICKUP_PT],
            PayloadKeys.REQ_DROPOFF_WINDOW_START: request_data[PayloadKeys.REQ_DROPOFF_WINDOW_START], 
            PayloadKeys.REQ_DROPOFF_WINDOW_END:   request_data[PayloadKeys.REQ_DROPOFF_WINDOW_END],
            PayloadKeys.REQ_DROPOFF_PT:           request_data[PayloadKeys.REQ_DROPOFF_PT],
        }
        return request

    @staticmethod
    def _is_canonical_structure(data: dict) -> bool:
        """
        Detects whether the JSON already matches the canonical structure in the 'wilson' format.
        """
        driver_runs = data.get(PayloadKeys.DRIVERS, [])
        if len(driver_runs) == 0:
            return False
        return PayloadKeys.DRIVER_STATE in driver_runs[0]

    @staticmethod
    def _normalize_matrix_key(data: dict) -> dict:
        """
        Normalize matrix keys to canonical `travel_time_matrix`.

        Rules:
        - keep existing `travel_time_matrix`
        - otherwise rename legacy `time_matrix`
        - otherwise add `travel_time_matrix = None`
        """
        normalized = copy.deepcopy(data)
        if PayloadKeys.TIME_MATRIX in normalized:
            return normalized
        if "time_matrix" in normalized:
            normalized[PayloadKeys.TIME_MATRIX] = normalized.pop("time_matrix")
        else:
            normalized[PayloadKeys.TIME_MATRIX] = None
        return normalized

    @staticmethod
    def normalize_to_canonical(data: dict) -> dict:
        """
        Converts the newer JSON structure from 'chattanooga' into the expected structure of 'wilson'. 
        For structural differences, see 'Documentation.md'. The changes are only additions and no prior information is lost.
        """
        if PayloadParser._is_canonical_structure(data):
            # Keep Wilson `driver_runs.state` and `manifest` untouched.
            return PayloadParser._normalize_matrix_key(data)

        normalized = copy.deepcopy(data)

        depot_loc = normalized[PayloadKeys.DEPOT][PayloadKeys.DEPOT_PT]

        new_driver_runs = []
        for run in normalized[PayloadKeys.DRIVERS]:
            # Chattanooga stores state fields directly at run level.
            run_state = run
            state = {
                # copy old state
                PayloadKeys.DRIVER_STATE_RUN_ID: run_state[PayloadKeys.DRIVER_STATE_RUN_ID],
                PayloadKeys.DRIVER_STATE_START_TIME: run_state[PayloadKeys.DRIVER_STATE_START_TIME],
                PayloadKeys.DRIVER_STATE_END_TIME: run_state[PayloadKeys.DRIVER_STATE_END_TIME],
                PayloadKeys.DRIVER_STATE_AM_CAP: run_state[PayloadKeys.DRIVER_STATE_AM_CAP],
                PayloadKeys.DRIVER_STATE_WC_CAP: run_state[PayloadKeys.DRIVER_STATE_WC_CAP],
                # injected defaults
                PayloadKeys.DRIVER_STATE_LOC_SERV: 0,
                PayloadKeys.DRIVER_STATE_DT_SEC: 0,
                # initialize location at depot
                PayloadKeys.DRIVER_STATE_LOC: {
                    "lat": depot_loc["lat"],
                    "lon": depot_loc["lon"],
                }
            }
            new_driver_runs.append({
                PayloadKeys.DRIVER_STATE: state,
                PayloadKeys.DRIVER_MANIFEST: []})

        normalized[PayloadKeys.DRIVERS] = new_driver_runs

        normalized = PayloadParser._normalize_matrix_key(normalized)

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
        requests = copy.deepcopy(data.get(PayloadKeys.REQUESTS, []))

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
        vehicles = data[PayloadKeys.DRIVERS][:max_vehicles]
        depot = data[PayloadKeys.DEPOT]

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

    
    