import logging
import pandas as pd
from structure.request import Request
from structure.node import Node
from handlers.network_handler import NetworkHandler
from dateutil import parser
from datetime import datetime
from datetime import timedelta
from multiprocessing.pool import ThreadPool

PICKUP_TIME = 'tpep_pickup_datetime'
ID = 'id'
PICKUP_LAT = 'pickup_latitude'
PICKUP_LON = 'pickup_longitude'
DROPOFF_LAT = 'dropoff_latitude'
DROPOFF_LON = 'dropoff_longitude'
DWELL_PICKUP = 'dwell_pickup'
DWELL_ALIGHT = 'dwell_alight'

class RequestHandler:
    def __init__(self, filename, maximum_detour, maximum_waiting):
        self.filename = filename
        self.maximum_detour = maximum_detour
        self.maximum_waiting = maximum_waiting
        dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
        self.requests = pd.read_csv(filename,parse_dates=[PICKUP_TIME],date_parser=dateparse).sort_values(by = [PICKUP_TIME])
        with ThreadPool(10) as pool:
            for index,_ in self.requests.iterrows():
                pool.apply_async(self.update_request_location,args=(index,))
            pool.close()
            pool.join()

        self.count = self.requests.shape[0]
        self.next_index = 0
        logging.info('Total No of requests: {0}'.format(self.count))

    def update_request_location(self,index):
        row = self.requests.iloc[index]
        lat,lon = NetworkHandler.get_nearest_node(row[PICKUP_LAT],row[PICKUP_LON])
        self.requests.at[index,PICKUP_LAT] = lat
        self.requests.at[index,PICKUP_LON] = lon

        lat,lon = NetworkHandler.get_nearest_node(row[DROPOFF_LAT],row[DROPOFF_LON])
        self.requests.at[index,DROPOFF_LAT] = lat
        self.requests.at[index,DROPOFF_LON] = lon

    def earliest_start_time(self):
        start_time = self.get_request_by_iloc(0).pick_up_time
        logging.debug('Start time of first request: {0}'.format(start_time))
        return start_time

    def latest_start_time(self):
        start_time = self.get_request_by_iloc(self.count-1).pick_up_time
        logging.debug('Start time of last request: {0}'.format(start_time))
        return start_time

    def get_request(self,request_data):
        origin = Node(request_data[PICKUP_LAT],request_data[PICKUP_LON])
        destination = Node(request_data[DROPOFF_LAT],request_data[DROPOFF_LON])
        id = request_data[ID]
        pick_up_time = request_data[PICKUP_TIME]
        latest_pick_up_time = pick_up_time + timedelta(seconds=self.maximum_waiting)
        travel_time = NetworkHandler.travel_time(origin,destination)
        duration = travel_time + self.maximum_detour
        latest_arrival_time = pick_up_time + timedelta(seconds=duration)
        dwell_pickup = int(request_data[DWELL_PICKUP])
        dwell_alight = int(request_data[DWELL_ALIGHT])
        return Request(id,pick_up_time,latest_pick_up_time,latest_arrival_time,origin,destination,dwell_pickup,dwell_alight)

    def get_request_by_iloc(self,iloc):
        request_data = self.requests.iloc[iloc]
        return self.get_request(request_data)

    def get_batch(self,end_time,max_batch_size):
        batch = []
        ending_index = min(self.next_index+max_batch_size,self.requests.shape[0]-1)
        for _, row in self.requests.iloc[self.next_index:ending_index].iterrows():
            request = self.get_request(row)
            if request.pick_up_time > end_time:
                break
            batch.append(request)
            self.next_index+=1
        time_of_next_request = self.requests.iloc[self.next_index][PICKUP_TIME]
        if time_of_next_request <= end_time and len(batch) > 0:
            end_time = min(end_time,batch[-1].pick_up_time)
        print(end_time,len(batch))
        return batch,end_time
    
    def get_lookahead_trips(self,end_time,rh_factor,batch_interval):
        batch = []
        horizen_end_time = end_time + rh_factor*batch_interval
        for _, row in self.requests.iloc[self.next_index:].iterrows():
            request = self.get_request(row)
            if request.pick_up_time > horizen_end_time or request.pick_up_time < end_time:
                break
            batch.append(request)
        return batch
    
    def unique_nodes(self):
        return self.requests.origin.unique()
    
    def get_all_nodes(self,round_at):
        coordinates = {}
        nodes = []
        for _,request_data in self.requests.iterrows():
            lat,lon = round(request_data['pickup_latitude'],round_at),round(request_data['pickup_longitude'],round_at)
            coordinates[(lat,lon)] = None
            lat,lon = round(request_data['dropoff_latitude'],round_at),round(request_data['dropoff_longitude'],round_at)
            coordinates[(lat,lon)] = None
        for key in coordinates:
            nodes.append(Node(key[0],key[1]))
        return nodes
