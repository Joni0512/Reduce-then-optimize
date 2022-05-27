import logging
import pandas as pd
from structure.request import Request
from dateutil import parser
from datetime import datetime
from datetime import timedelta

PICKUP_TIME = 'tpep_pickup_datetime'
ORIGIN = 'origin'
DEST = 'dest'
ID = 'id'

class RequestHandler:
    def __init__(self, filename, max_wait_time, trip_lenghen_factor):
        self.filename = filename
        self.max_wait_time = max_wait_time
        self.trip_lenghen_factor = trip_lenghen_factor
        dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
        self.requests = pd.read_csv(filename,parse_dates=[PICKUP_TIME],date_parser=dateparse).sort_values(by = [PICKUP_TIME])
        self.count = self.requests.shape[0]
        logging.info('Total No of requests: {0}'.format(self.count))

    def earliest_start_time(self,network_handler):
        start_time = self.get_request_by_iloc(network_handler,0).pick_up_time
        logging.debug('Start time of first request: {0}'.format(start_time))
        return start_time

    def latest_start_time(self,network_handler):
        start_time = self.get_request_by_iloc(network_handler,self.count-1).pick_up_time
        logging.debug('Start time of last request: {0}'.format(start_time))
        return start_time

    def get_request(self,network_handler,request_data):
        origin = int(request_data[ORIGIN])
        destination = int(request_data[DEST])
        id = request_data[ID]
        # pick_up_time = parser.parse(request_data[PICKUP_TIME])
        pick_up_time = request_data[PICKUP_TIME]
        latest_arrival_time = pick_up_time + timedelta(seconds=self.max_wait_time+int(self.trip_lenghen_factor*network_handler.travel_time(origin, destination)))
        return Request(id,pick_up_time,latest_arrival_time,origin,destination)

    def get_request_by_iloc(self,network_handler,iloc):
        request_data = self.requests.iloc[iloc]
        return self.get_request(network_handler,request_data)

    def get_batch(self,network_handler,start_time,end_time):
        batch = []
        for _, row in self.requests[(self.requests[PICKUP_TIME]>=start_time) & (self.requests[PICKUP_TIME]<end_time)].iterrows():
            # print(type(row))
            request = self.get_request(network_handler,row)
            batch.append(request)
        return batch
