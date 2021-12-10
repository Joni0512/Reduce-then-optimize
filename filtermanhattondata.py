import networkx as nx
import pickle
import pandas as pd
# import geopy.distance as dist
import numpy as np
import sys
from shapely.geometry import Point
import osmnx as ox
import time

DROPOFF_LAT = 'dropoff_latitude'
DROPOFF_LONG = 'dropoff_longitude'
PICKUP_LAT = 'pickup_latitude'
PICKUP_LONG = 'pickup_longitude'
PICKUP_TIME = 'tpep_pickup_datetime'
TRIP_DISTANCE = 'trip_distance'
ORIGIN = 'origin'
DEST = 'dest'

def getData(filename):
    return pd.read_csv(filename,parse_dates=True, usecols=[PICKUP_TIME, PICKUP_LONG, PICKUP_LAT, DROPOFF_LONG, DROPOFF_LAT, TRIP_DISTANCE]).sort_values(by = [PICKUP_TIME])

# def boundToNYC(data):
#     return data.loc[(data[PICKUP_LAT] >= 40.65) & 
#                                 (data[PICKUP_LAT] <= 40.9) &
#                                 (data[PICKUP_LONG] >= -74.05) &
#                                 (data[PICKUP_LONG] <= -73.9) &
#                                 (data[DROPOFF_LAT] >= 40.65) & 
#                                 (data[DROPOFF_LAT] <= 40.9) &
#                                 (data[DROPOFF_LONG] >= -74.05) &
#                                 (data[DROPOFF_LONG] <= -73.9)]

# def filterManhatton(data,network,cutoff):
#     selected_trips = []
#     count = data.shape[0]
#     data.insert(6,ORIGIN,np.zeros(count))
#     data.insert(7,DEST,np.zeros(count))
#     for index, row in data.iterrows():
#         close_pickup, distance = find_closest_node(network,row[PICKUP_LAT],row[PICKUP_LONG])
#         if distance < cutoff:
#             close_drop, distance = find_closest_node(network,row[DROPOFF_LAT],row[DROPOFF_LONG])
#             if distance < cutoff:
#                 selected_trips.append(index)
#                 data.at[index,ORIGIN] = close_pickup
#                 data.at[index,DEST] = close_drop
#     print("Selected: ",len(selected_trips))
#     return data.loc[selected_trips]

# def find_closest_node(network,lat,long):
#     closest_distance = -1
#     closest = None
#     coordinates = (lat,long)
#     for node in network.nodes():
#         node_info = network.nodes[node]
#         distance = abs(dist.geodesic(coordinates, (node_info['y'],node_info['x'])).m)
#         if closest_distance < 0 or distance < closest_distance:
#             closest = node
#             closest_distance = distance
#     return closest,closest_distance


city = ox.geocode_to_gdf('Manhattan, New York City, New York, USA')
geom = city.loc[0, 'geometry']

start = int(sys.argv[1])
end = int(sys.argv[2])



# print(start, end)
Manhattan_network = pickle.load(open("Manhattan_network.p", "rb"))
data = getData('yellow_tripdata_2015-01.csv')[start:end]
count = data.shape[0]
data.insert(6,ORIGIN,np.zeros(count))
data.insert(7,DEST,np.zeros(count))

st = time.time()
selected_trips = []
# count = 0
for index, row in data.iterrows():
        cord1 = (row[PICKUP_LONG],row[PICKUP_LAT])
        cord2 = (row[DROPOFF_LONG], row[DROPOFF_LAT])
        fit1 = geom.intersects(Point(cord1))
        fit2 = geom.intersects(Point(cord2))
        if fit1 and fit2:
            selected_trips.append(index)
            ori = ox.distance.nearest_nodes(Manhattan_network, row[PICKUP_LONG],row[PICKUP_LAT])
            des = ox.distance.nearest_nodes(Manhattan_network, row[DROPOFF_LONG], row[DROPOFF_LAT])
            if type(ori) == list:
                data.at[index,ORIGIN] = ori[0]
            else:
                data.at[index,ORIGIN] = ori
            if type(des) == list:    
                data.at[index,DEST] = des[0]
            else:  
                data.at[index,DEST] = des

# print(time.time()-st)
data.loc[selected_trips].to_csv("manhatton_demand_filtered_"+sys.argv[1]+"_"+sys.argv[2]+".csv")
# filterManhatton(boundToNYC(getData('yellow_tripdata_2015-01.csv'))[start:end],Manhattan_network,500).to_csv("manhatton_demand_"+sys.argv[1]+"_"+sys.argv[2]+".csv")
# print("Done")
