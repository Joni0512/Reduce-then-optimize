from datetime import datetime
from datetime import timedelta
from handlers.request_handler import RequestHandler
from handlers.network_handler import NetworkHandler
from handlers.vehicle_handler import VehicleHandler
from handlers.trip_handler import TripHandler

class OnlineRTVSolver:

    def __init__(self):
        self.ILP_SOLVER_TIMEOUT = 1200
        self.PENALTY = 1000000 # penalty for not serving a trip
        self.SHAREABLE_COST_FACTOR = 1
        self.MAX_CARDINALITY = 8
        self.MAX_THREAD_CNT = 500
        self.REBALANCING = False
        self.RH_FACTOR = 0
        self.DWELL_PICKUP = 180
        self.DWELL_ALIGHT = 60
    
    def build_request_from_driver_manifest(manifest,pick_up_index):
        stop = manifest[pick_up_index]
        booking_id = stop['booking_id']
        for drop_off_stop in manifest[pick_up_index+1:]:
            if drop_off_stop['booking_id'] == booking_id:
                return OnlineRTVSolver.build_request(stop,drop_off_stop)
            
    def build_request_from_manifest(manifest,drop_off_stop):
        booking_id = drop_off_stop['booking_id']
        for pick_up_stop in manifest:
            if pick_up_stop['booking_id'] == booking_id and pick_up_stop['action'] == 'pickup':
                return OnlineRTVSolver.build_request(pick_up_stop,drop_off_stop)
            
    def build_request(pick_up_stop,drop_off_stop):
        request = {'am': pick_up_stop['am'], 'wc': pick_up_stop['wc'], 'pickup_time_window_start': pick_up_stop['time_window_start'], 
        'pickup_time_window_end': pick_up_stop['time_window_end'], 'pickup_pt': pick_up_stop['loc'], 'booking_id': pick_up_stop['booking_id'],
        'dropoff_time_window_start': drop_off_stop['time_window_start'], 'dropoff_time_window_end': drop_off_stop['time_window_end'],
        'dropoff_pt': drop_off_stop['loc']
        }
        return request
            
    def solve_rtv(self, payload):
        NetworkHandler.init(False,payload=payload)
        active_requests_data = {}
        boarded_requests_data = {}

        manifest_sorted_by_vehicles = {}
        for stop in payload["manifests"]:
            run_id = stop["run_id"]
            if run_id not in manifest_sorted_by_vehicles:
                manifest_sorted_by_vehicles[run_id] = []
            manifest_sorted_by_vehicles[run_id].append(stop)


        for driver_run in payload["driver_runs"]:
            i = 0
            added_active_requests = []
            while i < len(driver_run['manifest']):
                stop = driver_run['manifest'][i]
                booking_id = stop['booking_id']
                if stop['action'] == 'pickup':
                    request = OnlineRTVSolver.build_request_from_driver_manifest(driver_run['manifest'],i)
                    if i == 0:
                        boarded_requests_data[booking_id] = request
                    else:
                        added_active_requests.append(booking_id)
                        active_requests_data[booking_id] = request
                else:
                    if booking_id not in active_requests_data:
                        run_id = stop["run_id"]
                        request = OnlineRTVSolver.build_request_from_manifest(manifest_sorted_by_vehicles[run_id],stop)
                        boarded_requests_data[booking_id] = request
                i+=1

        if 'requests' not in payload:
            request = OnlineRTVSolver.build_request(payload['pickup'],payload['dropoff'])
            payload['requests'] = [request]
            
        for req_id in active_requests_data:
            payload['requests'].append(active_requests_data[req_id])

        for req_id in boarded_requests_data:
            payload['requests'].append(boarded_requests_data[req_id])

        start_of_the_day = datetime.strptime(payload['date'], '%Y-%m-%d')
        request_handler = RequestHandler(payload, start_of_the_day, self.DWELL_PICKUP, self.DWELL_ALIGHT)
        temp_batch = request_handler.get_all_requests()
        batch = []
        active_requests = {}
        boarded_requests = {}
        for req in temp_batch:
            req_id = req.id
            if req_id in boarded_requests_data:
                boarded_requests[req_id] = req
            else:
                if req_id in active_requests_data:
                    active_requests[req_id] = req
                batch.append(req)

        current_time = 3600*24
        earliest_start_time = current_time
        for driver_run in payload['driver_runs']:
            last_recorded_time = driver_run['state']['location_dt_seconds']
            start_time = driver_run['state']['start_time']
            earliest_start_time = min(earliest_start_time,start_time)
            if last_recorded_time > start_time:
                current_time = min(current_time,last_recorded_time)
        if current_time == 3600*24:
            current_time = earliest_start_time

        current_time = start_of_the_day +timedelta(seconds=int(current_time))
        iteration = 0
        boarded_trips = TripHandler.create_trip_for_picked_requests(boarded_requests,iteration)

        vehicle_handler = VehicleHandler(payload,None,start_of_the_day)
        vehicle_handler.add_manifest_to_vehicles(current_time,start_of_the_day,payload,boarded_requests,boarded_trips,self.DWELL_ALIGHT, self.DWELL_PICKUP)

        iteration+=1
        trip_handler = TripHandler(current_time,vehicle_handler.vehicles,batch, active_requests, iteration, self.ILP_SOLVER_TIMEOUT,self.PENALTY,self.MAX_CARDINALITY,self.MAX_THREAD_CNT,self.SHAREABLE_COST_FACTOR,self.REBALANCING)
        for vehicle_id in trip_handler.vehicle_assignment:
            vehicle = vehicle_handler.vehicles[vehicle_id]
            trips = trip_handler.vehicle_assignment[vehicle_id]
            VehicleHandler.add_new_trips(current_time, vehicle, trips, add=True)

        # create updated driver runs
        updated_driver_runs = []
        for driver_run in payload["driver_runs"]:
            new_state = driver_run["state"]
            manifest = driver_run["manifest"]
            new_manifest = []
            current_order = new_state['locations_already_serviced']
            if len(manifest) > 0:
                new_manifest.append(manifest[0])
                current_order=manifest[0]["order"]
            vehicle = vehicle_handler.vehicles[new_state["run_id"]]
            new_manifest.extend(VehicleHandler.get_manifest(vehicle,current_order,start_of_the_day))
            new_state["total_locations"] = new_state["total_locations"] + len(new_manifest) - len(manifest)
            new_driver_run = {"state":new_state,"manifest":new_manifest}
            updated_driver_runs.append(new_driver_run)

        return updated_driver_runs
