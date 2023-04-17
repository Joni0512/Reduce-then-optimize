import logging
import pandas as pd
from structure.request import Request
from handlers.network_handler import NetworkHandler
from dateutil import parser
from datetime import datetime
from datetime import timedelta

PICKUP_TIME = 'tpep_pickup_datetime'
ORIGIN = 'origin'
DEST = 'dest'
ID = 'id'

class RequestHandler:
    def __init__(self, filename, minimum_trip_duration, trip_lenghen_factor):
        self.filename = filename
        self.minimum_trip_duration = minimum_trip_duration
        self.trip_lenghen_factor = trip_lenghen_factor
        dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
        self.requests = pd.read_csv(filename,parse_dates=[PICKUP_TIME],date_parser=dateparse).sort_values(by = [PICKUP_TIME])
        self.count = self.requests.shape[0]
        self.next_index = 0
        logging.info('Total No of requests: {0}'.format(self.count))

    def earliest_start_time(self):
        start_time = self.get_request_by_iloc(0).pick_up_time
        logging.debug('Start time of first request: {0}'.format(start_time))
        return start_time

    def latest_start_time(self):
        start_time = self.get_request_by_iloc(self.count-1).pick_up_time
        logging.debug('Start time of last request: {0}'.format(start_time))
        return start_time

    def get_request(self,request_data):
        origin = int(request_data[ORIGIN])
        destination = int(request_data[DEST])
        id = request_data[ID]
        pick_up_time = request_data[PICKUP_TIME]
        travel_time = NetworkHandler.travel_time(origin,destination)
        duration = int((1+self.trip_lenghen_factor*(max(0.5/self.trip_lenghen_factor,1-travel_time/3600)))*travel_time)
        duration = max(duration,self.minimum_trip_duration)
        latest_arrival_time = pick_up_time + timedelta(seconds=duration)
        return Request(id,pick_up_time,latest_arrival_time,origin,destination)

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
    
    def unique_nodes(self):
        return self.requests.origin.unique()
