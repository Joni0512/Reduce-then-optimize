class TripCost:
    """
    TODO TripCost 
    """
    def __init__(self, trip_no, vehicle_id, cost, sequence):
        self.trip_no = trip_no
        self.vehicle_id = vehicle_id
        self.cost = cost
        self.sequence = sequence

    def __str__(self):
        seq_str = "->".join([str(node) for node in self.sequence])
        return f"TripCost(t_no={self.trip_no}, v_id={self.vehicle_id}, cost={self.cost}, seq={seq_str})"