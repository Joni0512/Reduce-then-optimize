from typing import Optional
from rtv_solver.structure.vehicle_stop import VehicleStop
from rtv_solver.structure.sequence import StopSequence
from rtv_solver.structure.trip_insertion_plan import TripInsertionPlan

class TripCost:
    """manages keys for merging trip, vehicle and cost"""
    def __init__(self, trip_no, vehicle_id, cost, sequence, plan: TripInsertionPlan = None):
        self.trip_no: int = trip_no
        self.vehicle_id: int = vehicle_id
        self.cost: int = cost
        self.sequence = StopSequence(sequence)
        self.plan: Optional[TripInsertionPlan] = plan

    def simple_str(self):
        return f"TC({self.trip_no}, veh={self.vehicle_id}, cost={self.cost:.1f}, seq={self.sequence.simple_str()})"

    def get_request_order_str(self):
        """get quick string for all requests for easier debugging"""
        return self.sequence.request_str()

    def get_ordered_stop_sequence(self):
        """get the stop sequence in the order of the sequence"""
        ordered_stop_sequence = []
        for stop in self.sequence:
            # only get the first letter of the type string
            type = stop.type[0].upper()
            request_id = stop.get_requestID_stripped_iteration()
            ordered_stop_sequence.append(f"{type}{request_id}")
        return ordered_stop_sequence

    def get_ordered_request_ids(self):
        """get list of request ids in the order of the sequence"""
        return [stop.get_requestID_stripped_iteration() for stop in self.sequence if stop.type == VehicleStop.ACT_PICKUP]

    def __str__(self):
        return f"TripCost(t_no={self.trip_no}, v_id={self.vehicle_id}, cost={self.cost}, seq={self.sequence})"
    
    def __repr__(self):
        return (
            f"TripCost("
            f"trip_no={self.trip_no!r}, "
            f"vehicle_id={self.vehicle_id!r}, "
            f"cost={self.cost!r}, "
            f"sequence={self.sequence!r}, "
            f"plan={self.plan!r}"
            f")"
        )