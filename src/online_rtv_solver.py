from handlers.request_handler import RequestHandler
from handlers.network_handler import NetworkHandler
from handlers.vehicle_handler import VehicleHandler
from handlers.trip_handler import TripHandler
from handlers.payload_parser import PayloadParser

class OnlineRTVSolver:

    def __init__(self,server_url,SHAREABLE_COST_FACTOR=1,RTV_TIMEOUT=3000, LARGEST_TSP = 10):
        self.ILP_SOLVER_TIMEOUT = 120 # seconds
        self.RTV_TIMEOUT = RTV_TIMEOUT #seconds
        self.PENALTY = 1000000 # penalty for not serving a trip
        self.SHAREABLE_COST_FACTOR = SHAREABLE_COST_FACTOR
        self.MAX_CARDINALITY = 4
        self.MAX_THREAD_CNT = 64
        self.REBALANCING = False
        self.RH_FACTOR = 1
        self.DWELL_PICKUP = 180
        self.DWELL_ALIGHT = 60
        self.LARGEST_TSP = LARGEST_TSP
        self.server_url = server_url
            
    def solve_rtv(self, current_time, payload):
        NetworkHandler.init(True, self.server_url)
        payload_object = PayloadParser.get_payload_object(payload)
        request_handler = RequestHandler(payload_object.requests, self.DWELL_PICKUP, self.DWELL_ALIGHT)
        temp_batch = request_handler.get_all_requests()
        batch = []
        active_requests = {}
        boarded_requests = {}
        for req in temp_batch:
            req_id = req.id
            if req_id in payload_object.boarded_requests:
                boarded_requests[req_id] = req
            else:
                if req_id in payload_object.active_requests:
                    active_requests[req_id] = req
                batch.append(req)

        iteration = 0
        boarded_trips = TripHandler.create_trip_for_picked_requests(boarded_requests,iteration)

        vehicle_handler = VehicleHandler(payload_object.depot, payload_object.driver_runs,None,LARGEST_TSP=self.LARGEST_TSP)
        vehicle_handler.add_manifest_to_vehicles(payload_object.driver_runs,boarded_requests,boarded_trips,self.DWELL_ALIGHT, self.DWELL_PICKUP)

        NetworkHandler.initialize_travel_time_matrix()
        iteration+=1
        trip_handler = TripHandler(current_time,vehicle_handler.vehicles,batch, active_requests, iteration, self.ILP_SOLVER_TIMEOUT,self.PENALTY,self.MAX_CARDINALITY,self.MAX_THREAD_CNT,self.SHAREABLE_COST_FACTOR,self.REBALANCING,self.RTV_TIMEOUT)
        for vehicle_id in trip_handler.vehicle_assignment:
            vehicle = vehicle_handler.vehicles[vehicle_id]
            trips = trip_handler.vehicle_assignment[vehicle_id]
            VehicleHandler.add_new_trips(current_time, vehicle, trips, add=True)

        # create updated driver runs
        updated_driver_runs = []
        for driver_run in payload_object.driver_runs:
            state = driver_run[PayloadParser.DRIVER_STATE]
            manifest = driver_run[PayloadParser.DRIVER_MANIFEST]
            current_order = state[PayloadParser.DRIVER_STATE_LOC_SERV]
            new_manifest = manifest[:current_order]
            vehicle = vehicle_handler.vehicles[state[PayloadParser.DRIVER_STATE_RUN_ID]]
            new_manifest.extend(VehicleHandler.get_manifest(vehicle,current_order))
            state[PayloadParser.DRIVER_STATE_T_LOCS] = len(new_manifest)
            new_driver_run = {PayloadParser.DRIVER_STATE:state,PayloadParser.DRIVER_MANIFEST:new_manifest}
            updated_driver_runs.append(new_driver_run)

        return updated_driver_runs #,trip_handler,vehicle_handler,request_handler,payload_object

    def simulate_manifest(self, current_time, date, driver_runs):
        NetworkHandler.init(True, self.server_url)
        start_of_the_day = datetime.strptime(date, '%Y-%m-%d')
        new_driver_runs = []
        for driver_run in driver_runs:
            state = driver_run[PayloadParser.DRIVER_STATE]
            current_order = state[PayloadParser.DRIVER_STATE_LOC_SERV]
            manifest = driver_run[PayloadParser.DRIVER_MANIFEST]
            next_immediate_time = state[PayloadParser.DRIVER_STATE_DT_SEC]
            next_immediate_loc = state[PayloadParser.DRIVER_STATE_LOC]
            
            if len(manifest) == current_order and next_immediate_time < current_time:
                next_immediate_time = current_time

            while len(manifest) > current_order and current_time >= manifest[current_order]["scheduled_time"]:
                next_stop = manifest[current_order]
                next_immediate_time = next_stop["scheduled_time"]
                next_immediate_loc = next_stop["loc"]

                if next_stop["action"] == "pickup":
                    next_immediate_time += self.DWELL_PICKUP
                else:
                    next_immediate_time += self.DWELL_ALIGHT
                current_order+=1
                if next_immediate_time > current_time:
                    break
                
            
            if len(manifest) > current_order and next_immediate_time < current_time:
                next_immediate_node = NetworkHandler.manifest_location(next_immediate_loc)
                target_node = NetworkHandler.manifest_location(manifest[current_order]["loc"])
                current_time_dt = start_of_the_day + timedelta(seconds=int(current_time))
                next_immediate_time_dt = start_of_the_day + timedelta(seconds=int(next_immediate_time))
                next_immediate_time_dt,next_immediate_node = NetworkHandler.get_current_location_time(next_immediate_node,target_node,next_immediate_time_dt,current_time_dt)
                next_immediate_time = (next_immediate_time_dt - start_of_the_day).seconds
                next_immediate_loc = {"lat":next_immediate_node.lat,"lon":next_immediate_node.lon}
            state[PayloadParser.DRIVER_STATE_DT_SEC] = next_immediate_time
            state[PayloadParser.DRIVER_STATE_LOC] = next_immediate_loc
            state[PayloadParser.DRIVER_STATE_LOC_SERV] = current_order
            new_driver_runs.append({PayloadParser.DRIVER_STATE:state,PayloadParser.DRIVER_MANIFEST:manifest})
        return new_driver_runs
