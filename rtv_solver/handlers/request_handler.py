import logging
import pandas as pd
from rtv_solver.structure.request import Request
from rtv_solver.structure.node import Node
from rtv_solver.handlers.network_handler import NetworkHandler
from dateutil import parser
from multiprocessing.pool import ThreadPool
from datetime import timedelta

class RequestHandler:
    # keys for request dictionary
    PICKUP_TIME = 'pickup_time_window_start'
    REQ_ID = 'id'
    PICKUP_LAT = 'pickup_latitude'
    PICKUP_LON = 'pickup_longitude'
    DROPOFF_LAT = 'dropoff_latitude'
    DROPOFF_LON = 'dropoff_longitude'
    DWELL_PICKUP = 'dwell_pickup'
    DWELL_ALIGHT = 'dwell_alight'
    PICKUP_WINDOW_END = 'pickup_time_window_end'
    ARRIVAL_WINDOW_START = 'dropoff_time_window_start'
    ARRIVAL_WINDOW_END = 'dropoff_time_window_end'
    PICKUP_NODE_ID = 'pickup_node_id'
    DROPOFF_NODE_ID = 'dropoff_node_id'

    def __init__(self, request_data, dwell_pickup, dwell_alight):
        """create a sorted list of all requests in a pd.dataframe """
        requests = [self.build_request(req, dwell_pickup, dwell_alight) for req in request_data]  
        self.requests = pd.DataFrame(requests).astype({RequestHandler.REQ_ID: 'string'}).sort_values(by = [RequestHandler.PICKUP_TIME])
        self.requests.drop_duplicates(subset=RequestHandler.REQ_ID, keep="first")

        self.count = self.requests.shape[0]
        self.next_index = 0
        logging.info('Total No of requests: {0}'.format(self.count))

    @staticmethod
    def build_request(req, dwell_pickup, dwell_alight):
        # simplified code to build a single request dictionary from the raw request data
        pickup = req['pickup_pt']
        dropoff = req['dropoff_pt']

        pickup_lat, pickup_lon = pickup['lat'], pickup['lon']
        dropoff_lat, dropoff_lon = dropoff['lat'], dropoff['lon']

        return {
            RequestHandler.REQ_ID: req['booking_id'],

            RequestHandler.PICKUP_LAT: pickup_lat,
            RequestHandler.PICKUP_LON: pickup_lon,
            RequestHandler.PICKUP_NODE_ID: NetworkHandler.get_next_node_id(pickup_lat, pickup_lon),

            RequestHandler.DROPOFF_LAT: dropoff_lat,
            RequestHandler.DROPOFF_LON: dropoff_lon,
            RequestHandler.DROPOFF_NODE_ID: NetworkHandler.get_next_node_id(dropoff_lat, dropoff_lon),

            RequestHandler.PICKUP_TIME: req[RequestHandler.PICKUP_TIME],
            RequestHandler.PICKUP_WINDOW_END: req[RequestHandler.PICKUP_WINDOW_END],
            RequestHandler.ARRIVAL_WINDOW_START: req[RequestHandler.ARRIVAL_WINDOW_START],
            RequestHandler.ARRIVAL_WINDOW_END: req[RequestHandler. ARRIVAL_WINDOW_END],

            'am': req['am'],
            'wc': req['wc'],
            RequestHandler.DWELL_PICKUP: dwell_pickup,
            RequestHandler.DWELL_ALIGHT: dwell_alight,
        }


    def update_request_location(self,index):
        row = self.requests.iloc[index]
        lat,lon = NetworkHandler.get_nearest_node(row[RequestHandler.PICKUP_LAT],row[RequestHandler.PICKUP_LON])
        self.requests.at[index,RequestHandler.PICKUP_LAT] = lat
        self.requests.at[index,RequestHandler.PICKUP_LON] = lon

        lat,lon = NetworkHandler.get_nearest_node(row[RequestHandler.DROPOFF_LAT],row[RequestHandler.DROPOFF_LON])
        self.requests.at[index,RequestHandler.DROPOFF_LAT] = lat
        self.requests.at[index,RequestHandler.DROPOFF_LON] = lon
    
    def earliest_start_time(self):
        start_time = self.get_request_by_iloc(0).pick_up_time
        logging.debug('Start time of first request: {0}'.format(start_time))
        return start_time

    def latest_start_time(self):
        start_time = self.get_request_by_iloc(self.count-1).pick_up_time
        logging.debug('Start time of last request: {0}'.format(start_time))
        return start_time

    @staticmethod
    def get_request(request_data):
        pickup_node_id = request_data.get(RequestHandler.PICKUP_NODE_ID)
        dropoff_node_id = request_data.get(RequestHandler.DROPOFF_NODE_ID)

        origin = Node(
            request_data[RequestHandler.PICKUP_LAT],
            request_data[RequestHandler.PICKUP_LON],
            pickup_node_id,
        )
        destination = Node(
            request_data[RequestHandler.DROPOFF_LAT],
            request_data[RequestHandler.DROPOFF_LON],
            dropoff_node_id,
        )

        request_id = request_data[RequestHandler.REQ_ID]

        pickup_time = request_data[RequestHandler.PICKUP_TIME]
        latest_pickup_time = request_data[RequestHandler.PICKUP_WINDOW_END]
        earliest_arrival_time = request_data[RequestHandler.ARRIVAL_WINDOW_START]
        latest_arrival_time = request_data[RequestHandler.ARRIVAL_WINDOW_END]

        dwell_pickup = int(request_data[RequestHandler.DWELL_PICKUP])
        dwell_alight = int(request_data[RequestHandler.DWELL_ALIGHT])
        am_capacity = request_data['am']
        wc_capacity = request_data['wc']

        return Request(
            request_id,
            am_capacity,
            wc_capacity,
            pickup_time,
            latest_pickup_time,
            earliest_arrival_time,
            latest_arrival_time,
            origin,
            destination,
            dwell_pickup,
            dwell_alight,
        )

    def get_request_by_iloc(self, iloc):
        request_data = self.requests.iloc[iloc]
        return self.get_request(request_data)

    def get_batch(self, end_time, max_batch_size):
        batch = []
        ending_index = min(self.next_index+max_batch_size,self.requests.shape[0])
        for _, row in self.requests.iloc[self.next_index:ending_index].iterrows():
            request = self.get_request(row)
            if request.pick_up_time > end_time:
                break
            batch.append(request)
            self.next_index+=1
        time_of_next_request = self.requests.iloc[min(self.next_index,self.requests.shape[0]-1)][RequestHandler.PICKUP_TIME]
        if time_of_next_request <= end_time and len(batch) > 0:
            end_time = min(end_time, batch[-1].pick_up_time)
        print("T:", end_time, "batch:",len(batch))
        return batch, end_time
    
    def get_lookahead_trips(self,end_time,rh_factor,batch_interval:timedelta):
        batch = []
        horizen_end_time = end_time + rh_factor * batch_interval.total_seconds()
        for _, row in self.requests.iloc[self.next_index:].iterrows():
            request = self.get_request(row)
            if request.pick_up_time > horizen_end_time or request.pick_up_time < end_time:
                break
            batch.append(request)
        return batch

    def get_all_requests(self):
        # TODO why do we not just return the dataframe
        batch = []
        for _, row in self.requests.iterrows():
            request = self.get_request(row)
            batch.append(request)
        return batch
    
    def unique_nodes(self):
        return self.requests.origin.unique()
    
    def get_all_nodes(self,round_at):
        coordinates = {}
        nodes = []
        for _,request_data in self.requests.iterrows():
            lat,lon = round(request_data[RequestHandler.PICKUP_LAT],round_at),round(request_data[RequestHandler.PICKUP_LON],round_at)
            coordinates[(lat,lon)] = None
            lat,lon = round(request_data[RequestHandler.DROPOFF_LAT],round_at),round(request_data[RequestHandler.DROPOFF_LON],round_at)
            coordinates[(lat,lon)] = None
        for key in coordinates:
            nodes.append(Node(key[0],key[1]))
        return nodes
