from typing import List
from dataclasses import dataclass, asdict

from rtv_solver.schema.payload_keys import PayloadKeys
from rtv_solver.structure.node import Node

# DO NOT CHANGE THE VARIABLE NAMES FOR ANY VARIABLES OF THE DATACLASSES BELOW AS IT COULD BREAK THE CODE WHERE WE HAVE NOT YET UPDATED THE BASIS OF THE DICTS

@dataclass
class State:
    """simple dataclass as defined in the dictionary as the basis of this software"""
    run_id: int
    start_time: float
    end_time: float
    am_capacity: int
    wc_capacity: int
    locations_already_serviced: int
    location_dt_seconds: int
    total_locations: int
    loc: Node

    @classmethod
    def from_dict(cls, data: dict) -> 'State':
        return cls(
            run_id                      = data[PayloadKeys.DRIVER_STATE_RUN_ID],
            start_time                  = data[PayloadKeys.DRIVER_STATE_START_TIME],
            end_time                    = data[PayloadKeys.DRIVER_STATE_END_TIME],
            am_capacity                 = data[PayloadKeys.DRIVER_STATE_AM_CAP],
            wc_capacity                 = data[PayloadKeys.DRIVER_STATE_WC_CAP],
            locations_already_serviced  = data[PayloadKeys.DRIVER_STATE_LOC_SERV],
            location_dt_seconds         = data[PayloadKeys.DRIVER_STATE_DT_SEC],
            total_locations             = data.get(PayloadKeys.DRIVER_STATE_T_LOCS, 0), # not sure it exists always?
            loc                    = Node.from_dict(data[PayloadKeys.DRIVER_STATE_LOC])
        )
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ManifestEntry:
    """
    simple dataclass as defined in the dictionary as the basis of this software
    """
    run_id: int
    booking_id: int
    order: int
    action: List[str]  # e.g., ["pickup", "dropoff"]
    loc: Node
    am: int # wheelchair
    wc: int # ambulatory
    scheduled_time: int
    time_window_start: int
    time_window_end: int

    @classmethod
    def from_dict(cls, data: dict) -> 'ManifestEntry':
        return cls(
            run_id              = data[PayloadKeys.MANIFEST_RUN_ID],
            booking_id          = data[PayloadKeys.MANIFEST_BOOKING_ID],
            order               = data[PayloadKeys.MANIFEST_ORDER],
            action              = data[PayloadKeys.MANIFEST_ACTION],
            loc                 = Node.from_dict(data[PayloadKeys.MANIFEST_LOC]),
            am                  = data[PayloadKeys.MANIFEST_AMBULATORY],
            wc                  = data[PayloadKeys.MANIFEST_WHEELCHAIR],
            scheduled_time      = data[PayloadKeys.MANIFEST_SCHED_TIME],
            time_window_start   = data[PayloadKeys.MANIFEST_TIME_WINDOW_START],
            time_window_end     = data[PayloadKeys.MANIFEST_TIME_WINDOW_END]
        )
    
    def to_dict(self) -> dict:
        return asdict(self)
    

@dataclass
class DriverRun:
    """
    Simplifies data structure to use the data that is used int he 
    """
    state: State
    manifest: list[ManifestEntry]

    @classmethod
    def from_dict(cls, data: dict) -> 'DriverRun':
        state       = State.from_dict(data[PayloadKeys.DRIVER_STATE])
        manifest_entries = data[PayloadKeys.DRIVER_MANIFEST]
        manifest = []
        for idx, entry in enumerate(manifest_entries):
            manifest.append(ManifestEntry.from_dict(entry))
        return cls(state, manifest)
    
    def to_dict(self) -> dict:
        manifest_list = []
        for entry in self.manifest:
            manifest_list.append(entry.to_dict())
        return {
            PayloadKeys.DRIVER_STATE: self.state.to_dict(),
            PayloadKeys.DRIVER_MANIFEST: manifest_list
        }