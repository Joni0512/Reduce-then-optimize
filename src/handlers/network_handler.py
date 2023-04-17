import logging
from multiprocessing.sharedctypes import RawArray, RawValue
import ctypes
import numpy as np

class NetworkHandler:
    def init(base_directory,USE_REAL_DISTANCE):
        global times,predecessors,no_of_nodes,use_real_distance
        if USE_REAL_DISTANCE:
            global distances
            distances = np.genfromtxt(base_directory+'distance.csv', delimiter=',', dtype=np.int16) #
            distances = RawArray(np.ctypeslib.as_ctypes_type(distances.dtype), distances.flatten()) #
        times = np.genfromtxt(base_directory+'times.csv', delimiter=',', dtype=np.uint16)
        predecessors = np.genfromtxt(base_directory+'pred.csv', delimiter=',', dtype=np.uint16)
        
        no_of_nodes = RawValue(ctypes.c_uint, times.shape[0])
        times = RawArray(np.ctypeslib.as_ctypes_type(times.dtype), times.flatten())
        predecessors = RawArray(np.ctypeslib.as_ctypes_type(predecessors.dtype), predecessors.flatten())
        use_real_distance = RawValue(ctypes.c_bool, USE_REAL_DISTANCE)
        
        logging.info('Network size: {0}'.format(no_of_nodes))
        if USE_REAL_DISTANCE:
            return times,predecessors,distances,no_of_nodes,use_real_distance
        return times,predecessors,no_of_nodes,use_real_distance

    def get_location(source,destination):
        return (source-1)*no_of_nodes.value+destination-1

    def travel_time(source,destination):
        return times[NetworkHandler.get_location(source,destination)]

    def travel_distance(source,destination):
        if use_real_distance:
            return distances[NetworkHandler.get_location(source,destination)]
        return NetworkHandler.travel_time(source,destination)*(20/3.6)

    def predecessor(source,destination):
        return predecessors[NetworkHandler.get_location(source,destination)]

    def get_path(source,destination):
        path = [destination]
        current_target = destination
        while current_target != source:
            current_target = NetworkHandler.predecessor(source,current_target)
            path.append(current_target)
        path.reverse()
        return path

    def get_network_size():
        return no_of_nodes.value
