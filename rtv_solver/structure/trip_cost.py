from rtv_solver.structure.vehicle_stop import VehicleStop
from rtv_solver.structure.sequence import StopSequence

class TripCost:
    """manages keys for merging trip, vehicle and cost"""
    def __init__(self, trip_no, vehicle_id, cost, sequence):
        self.trip_no: int = trip_no
        self.vehicle_id: int = vehicle_id
        self.cost: int = cost
        self.sequence = StopSequence(sequence)

    def __str__(self):
        return f"TripCost(t_no={self.trip_no}, v_id={self.vehicle_id}, cost={self.cost}, seq={self.sequence})"
    
    def __repr__(self):
        return (
            f"TripCost("
            f"trip_no={self.trip_no!r}, "
            f"vehicle_id={self.vehicle_id!r}, "
            f"cost={self.cost!r}, "
            f"sequence={self.sequence!r}"
            f")"
        )