class TaxiOnlyAssignment():
    def __init__(self, vehicle_id):
        self.vehicle_id = vehicle_id
    
class AssignmentWithBus():
    def __init__(self, bus_trip, first_mile_vehicle=None, last_mile_vehicle=None):
        self.bus_trip = bus_trip
        self.first_mile_vehicle = first_mile_vehicle
        self.last_mile_vehicle = last_mile_vehicle
