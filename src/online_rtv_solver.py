from datetime import datetime
from datetime import timedelta
from handlers.request_handler import RequestHandler
from handlers.network_handler import NetworkHandler
from handlers.vehicle_handler import VehicleHandler
from handlers.trip_handler import TripHandler

active_requests_data = {}
boarded_requests_data = {}

class OnlineRTVSolver:
    ILP_SOLVER_TIMEOUT = 1200
    PENALTY = 1000000 # penalty for not serving a trip
    SHAREABLE_COST_FACTOR = 1
    MAX_CARDINALITY = 4
    MAX_THREAD_CNT = 500
    REBALANCING = False
    RH_FACTOR = 0
    DWELL_PICKUP = 180
    DWELL_ALIGHT = 60

    def __init__(self,server_url):
        NetworkHandler.init(server_url)
    
    def build_request(manifest,pick_up_index):
        stop = manifest[pick_up_index]
        booking_id = stop['booking_id']
        for drop_off_stop in manifest[pick_up_index+1:]:
            if drop_off_stop['booking_id'] == booking_id:
                request = {'am': stop['am'], 'wc': stop['wc'], 'pickup_time_window_start': stop['time_window_start'], 
                'pickup_time_window_end': stop['time_window_end'], 'pickup_pt': stop['loc'], 'booking_id': stop['booking_id'],
                'dropoff_time_window_start': drop_off_stop['time_window_start'], 'dropoff_time_window_end': drop_off_stop['time_window_end'],
                'dropoff_pt': drop_off_stop['loc']
                }
                return request
            
    def solve_rtv(self, payload):
        active_requests_data = {}
        boarded_requests_data = {}

        for driver_run in payload["driver_runs"]:
            locations_already_serviced = driver_run["state"]["locations_already_serviced"]
            i = 0
            while i < len(driver_run['manifest']):
                stop = driver_run['manifest'][i]
                booking_id = stop['booking_id']
                if stop['action'] == 'pickup':
                    request = OnlineRTVSolver.build_request(driver_run['manifest'],i)
                    if i <= locations_already_serviced:
                        boarded_requests_data[booking_id] = request
                    else:
                        active_requests_data[booking_id] = request
                else:
                    if i <= locations_already_serviced:
                        boarded_requests_data.pop(booking_id)
                i+=1

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

        current_time = start_of_the_day +timedelta(seconds=int(payload["current_time"]))
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
            i = 0
            current_order = 1
            while i <= new_state["locations_already_serviced"]:
                if i < len(manifest):
                    stop = manifest[i]
                    current_order = stop["order"]
                    new_manifest.append(stop)
                i+=1
            vehicle = vehicle_handler.vehicles[new_state["run_id"]]
            new_manifest.extend(VehicleHandler.get_manifest(vehicle,current_order,start_of_the_day))
            new_state["total_locations"] = len(new_manifest)
            new_driver_run = {"state":new_state,"manifest":new_manifest}
            updated_driver_runs.append(new_driver_run)

        result = {}
        for key in payload:
            result[key] = payload[key]
        result["driver_runs"] = updated_driver_runs
        return result
