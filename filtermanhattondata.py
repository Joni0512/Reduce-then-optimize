import datetime
import osmnx as ox
import numpy as np
import pickle
import pandas as pd
from shapely.geometry import Point
import sys

DROPOFF_LAT = 'dropoff_latitude'
DROPOFF_LONG = 'dropoff_longitude'
PICKUP_LAT = 'pickup_latitude'
PICKUP_LONG = 'pickup_longitude'
PICKUP_TIME = 'tpep_pickup_datetime'
TRIP_DISTANCE = 'trip_distance'
ORIGIN = 'origin'
DEST = 'dest'


start = sys.argv[1]
end = sys.argv[2]
print(start,end)
def getData(filename):
    return pd.read_csv(filename,parse_dates=True, usecols=[PICKUP_TIME, PICKUP_LONG, PICKUP_LAT, DROPOFF_LONG, DROPOFF_LAT, TRIP_DISTANCE]).sort_values(by = [PICKUP_TIME])

city = ox.geocode_to_gdf('Manhattan, New York City, New York, USA')
geom = city.loc[0, 'geometry']

Manhattan_network = pickle.load(open("manhatton/Manhattan_network.p", "rb"))
data = getData('yellow_tripdata_2015-01.csv')
data = data.loc[(data[PICKUP_TIME] >= start) & 
                                (data[PICKUP_TIME] < end)]
count = data.shape[0]
data.insert(6,ORIGIN,np.zeros(count))
data.insert(7,DEST,np.zeros(count))

selected_trips = []
for index, row in data.iterrows():
        cord1 = (row[PICKUP_LONG],row[PICKUP_LAT])
        cord2 = (row[DROPOFF_LONG], row[DROPOFF_LAT])
        fit1 = geom.intersects(Point(cord1))
        fit2 = geom.intersects(Point(cord2))
        if fit1 and fit2:
            ori = ox.distance.nearest_nodes(Manhattan_network, row[PICKUP_LONG],row[PICKUP_LAT])
            des = ox.distance.nearest_nodes(Manhattan_network, row[DROPOFF_LONG], row[DROPOFF_LAT])
            origin = None
            destination = None
            if type(ori) == list:
                origin = ori[0]
            else:
                origin = ori
            if type(des) == list:    
                destination = des[0]
            else:  
                destination = des
                
            data.at[index,ORIGIN] = origin
            data.at[index,DEST] = destination
            if origin != destination:
                selected_trips.append(index)

data.loc[selected_trips].to_csv("manhatton/input.csv")