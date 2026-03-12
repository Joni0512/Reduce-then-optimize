import copy

from rtv_solver.parser.base_parser import BaseParser
from rtv_solver.schema.payload_keys import PayloadKeys

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

        depot_loc = normalized[PayloadKeys.DEPOT][PayloadKeys.DEPOT_PT]

        new_driver_runs = []
        for run in normalized[PayloadKeys.DRIVERS]:
            state = {
                # copy old state
                PayloadKeys.DRIVER_STATE_RUN_ID: run[PayloadKeys.DRIVER_STATE_RUN_ID],
                PayloadKeys.DRIVER_STATE_START_TIME: run[PayloadKeys.DRIVER_STATE_START_TIME],
                PayloadKeys.DRIVER_STATE_END_TIME: run[PayloadKeys.DRIVER_STATE_END_TIME],
                PayloadKeys.DRIVER_STATE_AM_CAP: run[PayloadKeys.DRIVER_STATE_AM_CAP],
                PayloadKeys.DRIVER_STATE_WC_CAP: run[PayloadKeys.DRIVER_STATE_WC_CAP],
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

        return normalized