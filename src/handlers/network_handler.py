import logging
import pandas as pd
import numpy as np

class NetworkHandler:
    def __init__(self, base_directory):
        self.times = np.genfromtxt(base_directory+'times.csv', delimiter=',', dtype=np.int16)
        self.predecessors = np.genfromtxt(base_directory+'pred.csv', delimiter=',', dtype=np.int16)
        self.nodes = self.times.shape[0]
        
        print(self.predecessors[0,0])
        logging.info('Network size: {0}'.format(self.times.shape[0]))

    def travel_time(self,source,destination):
        return self.times[source-1,destination-1]

    def predecessor(self,source,destination):
        return self.predecessors[source-1,destination-1]
