from rtv_solver.structure.vehicle_stop import VehicleStop

class TripCost:
    """manages keys for merging trip, vehicle and cost"""
    def __init__(self, trip_no, vehicle_id, cost, sequence):
        self.trip_no: int = trip_no
        self.vehicle_id: int = vehicle_id
        self.cost: int = cost
        # TODO add a sequence object to combine interface between tripCost and vehicle
        self.sequence: list[VehicleStop] = sequence

    def __str__(self):
        return f"TripCost(t_no={self.trip_no}, v_id={self.vehicle_id}, cost={self.cost}, seq={TripCost.sequence_to_str(self.sequence)})"
    
    @staticmethod
    def sequence_to_str(sequence):
        """print out the sequence in order to make it readable in-line"""
        seq_str = "\n \t" + "\n\t".join([str(node) for node in sequence])
        return seq_str