import logging
import pandas as pd
from rtv_solver.structure.request import Request
from rtv_solver.structure.node import Node
from rtv_solver.handlers.network_handler import NetworkHandler
from rtv_solver.handlers.payload_parser import PayloadParser
from datetime import timedelta

class RequestHandler:
    """"""
    def __init__(self, request_data, dwell_pickup, dwell_alight):
        """create a sorted list of all requests in a pd.dataframe """
        requests = [self.build_request_dict(req, dwell_pickup, dwell_alight) for req in request_data] 
        len_requests_initial = len(requests)
        self.requests = pd.DataFrame(requests).astype({PayloadParser.REQ_BOOKING_ID: 'string'}).sort_values(by = [PayloadParser.REQ_PICKUP_WINDOW_START])
        self.requests.drop_duplicates(subset=PayloadParser.REQ_BOOKING_ID, keep="first")
        self.count = self.requests.shape[0]
        self.next_index = 0

        assert len_requests_initial == self.count, f"{len_requests_initial - self.count} requests were dropped as duplicates. Where did they come from?"      
        logging.info(f'{self.count} new and alreay assigned request(s) in payload')

    @staticmethod
    def build_request_dict(req, dwell_pickup, dwell_alight):
        # simplified code to build a single request dictionary from the raw request data
        pickup = req[PayloadParser.REQ_PICKUP_PT]
        pickup_lat, pickup_lon = pickup['lat'], pickup['lon']
        dropoff = req[PayloadParser.REQ_DROPOFF_PT]
        dropoff_lat, dropoff_lon = dropoff['lat'], dropoff['lon']

        return {
            PayloadParser.REQ_BOOKING_ID: req[PayloadParser.REQ_BOOKING_ID],

            PayloadParser.REQ_PICKUP_LAT: pickup_lat,
            PayloadParser.REQ_PICKUP_LON: pickup_lon,
            PayloadParser.REQ_PICKUP_NODE_ID: NetworkHandler.get_next_node_id(pickup_lat, pickup_lon),

            PayloadParser.REQ_DROPOFF_LAT: dropoff_lat,
            PayloadParser.REQ_DROPOFF_LON: dropoff_lon,
            PayloadParser.REQ_DROPOFF_NODE_ID: NetworkHandler.get_next_node_id(dropoff_lat, dropoff_lon),

            PayloadParser.REQ_PICKUP_WINDOW_START: req[PayloadParser.REQ_PICKUP_WINDOW_START],
            PayloadParser.REQ_PICKUP_WINDOW_END: req[PayloadParser.REQ_PICKUP_WINDOW_END],
            PayloadParser.REQ_DROPOFF_WINDOW_START: req[PayloadParser.REQ_DROPOFF_WINDOW_START],
            PayloadParser.REQ_DROPOFF_WINDOW_END: req[PayloadParser.REQ_DROPOFF_WINDOW_END],

            PayloadParser.REQ_AMBULATORY: req[PayloadParser.REQ_AMBULATORY],
            PayloadParser.REQ_WHEELCHAIR: req[PayloadParser.REQ_WHEELCHAIR],
            PayloadParser.REQ_DWELL_PICKUP: dwell_pickup,
            PayloadParser.REQ_DWELL_ALIGHT: dwell_alight,
        }

    def update_request_location(self,index):
        row = self.requests.iloc[index]
        lat,lon = NetworkHandler.get_nearest_node(row[PayloadParser.REQ_PICKUP_LAT],row[PayloadParser.REQ_PICKUP_LON])
        self.requests.at[index,PayloadParser.REQ_PICKUP_LAT] = lat
        self.requests.at[index,PayloadParser.REQ_PICKUP_LON] = lon

        lat,lon = NetworkHandler.get_nearest_node(row[PayloadParser.REQ_DROPOFF_LAT],row[PayloadParser.REQ_DROPOFF_LON])
        self.requests.at[index,PayloadParser.REQ_DROPOFF_LAT] = lat
        self.requests.at[index,PayloadParser.REQ_DROPOFF_LON] = lon
    
    def earliest_start_time(self):
        start_time = self.get_request_by_iloc(0).pick_up_time
        logging.debug('Start time of first request: {0}'.format(start_time))
        return start_time

    def latest_start_time(self):
        start_time = self.get_request_by_iloc(self.count-1).pick_up_time
        logging.debug('Start time of last request: {0}'.format(start_time))
        return start_time

    @staticmethod
    def get_request(request_data) -> Request:
        pickup_node_id = request_data.get(PayloadParser.REQ_PICKUP_NODE_ID)
        dropoff_node_id = request_data.get(PayloadParser.REQ_DROPOFF_NODE_ID)

        origin = Node(
            request_data[PayloadParser.REQ_PICKUP_LAT],
            request_data[PayloadParser.REQ_PICKUP_LON],
            pickup_node_id,
        )
        destination = Node(
            request_data[PayloadParser.REQ_DROPOFF_LAT],
            request_data[PayloadParser.REQ_DROPOFF_LON],
            dropoff_node_id,
        )

        return Request(
            request_data[PayloadParser.REQ_BOOKING_ID],
            request_data[PayloadParser.REQ_PICKUP_WINDOW_START],
            request_data[PayloadParser.REQ_PICKUP_WINDOW_END],
            request_data[PayloadParser.REQ_DROPOFF_WINDOW_START],
            request_data[PayloadParser.REQ_DROPOFF_WINDOW_END],
            origin,
            destination,
            int(request_data[PayloadParser.REQ_DWELL_PICKUP]),
            int(request_data[PayloadParser.REQ_DWELL_ALIGHT]),
            request_data[PayloadParser.REQ_AMBULATORY],
            request_data[PayloadParser.REQ_WHEELCHAIR],
        )

    def get_request_by_iloc(self, iloc):
        request_data = self.requests.iloc[iloc]
        return self.get_request(request_data)

    def get_batch(self, end_time, max_batch_size) -> tuple[list[Request], int]:
        batch = []
        ending_index = min(self.next_index+max_batch_size,self.requests.shape[0])
        for _, row in self.requests.iloc[self.next_index:ending_index].iterrows():
            request = self.get_request(row)
            if request.pick_up_time > end_time:
                break
            batch.append(request)
            self.next_index+=1
        time_of_next_request = self.requests.iloc[min(self.next_index,self.requests.shape[0]-1)][PayloadParser.REQ_PICKUP_WINDOW_START]
        if time_of_next_request <= end_time and len(batch) > 0:
            end_time = min(end_time, batch[-1].pick_up_time)
        logging.info(f"T: {end_time}, batch {len(batch)}")
        return batch, end_time
    
    def get_lookahead_trips(self, end_time, rh_factor, batch_interval: timedelta):
        batch = []
        horizen_end_time = end_time + rh_factor * batch_interval.total_seconds()
        for _, row in self.requests.iloc[self.next_index:].iterrows():
            request = self.get_request(row)
            if request.pick_up_time > horizen_end_time or request.pick_up_time < end_time:
                break
            batch.append(request)
        return batch

    def get_all_requests(self) -> list[Request]:
        batch = []
        for _, row in self.requests.iterrows():
            request = self.get_request(row)
            batch.append(request)
        return batch
    
    def get_unique_nodes(self):
        return self.requests.origin.unique()
    
    def get_all_nodes(self,round_at):
        coordinates = {}
        nodes = []
        for _,request_data in self.requests.iterrows():
            lat,lon = round(request_data[PayloadParser.REQ_PICKUP_LAT],round_at),round(request_data[PayloadParser.REQ_PICKUP_LON],round_at)
            coordinates[(lat,lon)] = None
            lat,lon = round(request_data[PayloadParser.REQ_DROPOFF_LAT],round_at),round(request_data[PayloadParser.REQ_DROPOFF_LON],round_at)
            coordinates[(lat,lon)] = None
        for key in coordinates:
            nodes.append(Node(key[0],key[1]))
        return nodes
