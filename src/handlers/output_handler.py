import logging
import time
from structure.assignment import TaxiOnlyAssignment

class OutputHandler:
    def __init__(self, output_directory):
        self.output_directory = output_directory
        self.request_count = 0
        self.unassigned_trip_count = 0
        self.taxi_only_trip_count = 0
        self.with_bus_trip_count = 0
        self.added_distance = 0

    def record_output(self,current_time,requests,trip_handler,total_time):
        with open(self.output_directory+"shareability.csv", 'a+') as shareability_file:
            shareability_file.write(",".join([str(i) for i in trip_handler.trip_sizes])+"\n")

        request_count = trip_handler.unassigned_trip_count + trip_handler.taxi_only_trip_count + trip_handler.with_one_bus_trip_count + trip_handler.with_two_bus_trip_count
        with open(self.output_directory+"summary.csv", 'a+') as summary_file:
            summary_file.write('{0},{1},{2},{3},{4},{5},{6},{7}\n'.format(current_time,total_time,request_count,trip_handler.unassigned_trip_count,trip_handler.taxi_only_trip_count,trip_handler.with_one_bus_trip_count,trip_handler.with_two_bus_trip_count,trip_handler.added_distance/1000))

        with open(self.output_directory+"assignment.csv", 'a+') as assignment_file:
            for request in requests:
                request_id = request.id
                if request_id in trip_handler.request_assignment:
                    assignment = trip_handler.request_assignment[request_id]
                    if isinstance(assignment,TaxiOnlyAssignment):
                        assignment_file.write('{0},ServedOnlyByTaxi,{1}\n'.format(request_id,assignment.vehicle_id))
                    else:
                        bus_trip = assignment.bus_trip
                        assignment_file.write('{0},ServedWithBus,{1},{2},{3},{4},{5},{6},{7},{8},{9},{10}\n'.format(request_id,":".join(bus_trip.bus_lines),bus_trip.pick_up_stop,bus_trip.transfer_point,bus_trip.destination_stop,assignment.first_mile_vehicle,assignment.last_mile_vehicle,bus_trip.leaving_time,bus_trip.arrival_time,bus_trip.arrival_at_transfer,bus_trip.departure_at_transfer))
                else:
                    assignment_file.write('{0}\n'.format(request_id))

    def record_vehicles(self,vehicle_locations,current_time):
        with open(self.output_directory+"vehicles.csv", 'a+') as location_file:
            sorted_vehicle_ids = list(vehicle_locations.keys())
            sorted_vehicle_ids.sort()
            location_file.write(",".join([str(vehicle_locations[vehicle_id]) for vehicle_id in sorted_vehicle_ids])+"\n")
        timestamp = time.mktime(current_time.timetuple())
        with open(self.output_directory+"vehicles_timestamp.csv", 'a+') as timestamp_file:
            timestamp_file.write(str(timestamp)+"\n")

    def record_completed_stops(self,completed_stops):
        with open(self.output_directory+"completed_stops.csv", 'a+') as location_file:
            for completed_stop in completed_stops:
                location_file.write(completed_stop.get_log()+"\n")
