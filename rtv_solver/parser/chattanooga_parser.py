import copy

from rtv_solver.parser.base_parser import BaseParser
from rtv_solver.handlers.payload_parser import PayloadParser

import pickle

class ChattanoogaParser(BaseParser):
    """
    Parser for Chattanooga PDPTW benchmark instances.
    """
    @staticmethod
    def parse_file(input_file: str) -> dict:
        """
        Converts the newer JSON structure from 'chattanooga' into the expected structure of 'wilson'. 
        For structural differences, see 'Documentation.md'. The changes are only additions and no prior information is lost. 
        """
        # FIXME generate travel time matrix for each of the request nodes (so not just the depot)
        with open(input_file, 'rb') as f:
            data = pickle.load(f)

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