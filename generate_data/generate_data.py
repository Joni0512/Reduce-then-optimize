import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx
import math
import numpy as np
from shapely.geometry import Point
import random
import datetime

BASE_OUTPUT_DIR = "data/"

def osmnx_routing_graph(addr='Manhattan, New York City, New York, USA', 
                        buffer_dist=500):
    """
    Uses OSMNX to get OSM routing graph for a city. Returns the routing graph (networkx)
    as well as the nodes and edges as pandas DataFrames. buffer_dist will expand the
    OSM network by buffer_dist (meters) in all directions.
    
    :param addr: str
    :param buffer_dist: int
    :return: networkx.MultiDiGraph, pandas.DataFrame, pandas.DataFrame
    """
    # get the OSM graph
    G = ox.graph_from_place(addr,
                            network_type='drive',
                            simplify=True,
                            truncate_by_edge=True,
                            retain_all=False,
                            buffer_dist=buffer_dist)
    G = ox.utils_graph.get_largest_component(G, strongly=True)

    # add edge speeds
    G = ox.speed.add_edge_speeds(G, fallback=40.2, precision=6)

    # add edge travel time
    G = ox.speed.add_edge_travel_times(G, precision=6)
    for n1, n2, k in G.edges(keys=True):
        G[n1][n2][k]['travel_time'] = math.ceil(G[n1][n2][k]['travel_time'])


    nodes, edges = ox.utils_graph.graph_to_gdfs(G)

    # format nodes
    nodes['osmid'] = nodes.index
    nodes.index = range(len(nodes))
    nodes['node_id'] = nodes.index
    nodes['lon'] = nodes['x']
    nodes['lat'] = nodes['y']
    nodes = nodes[['node_id', 'osmid', 'lat', 'lon']]
    nodes['node_id'] = nodes['node_id'].astype(int)
    nodes['osmid'] = nodes['osmid'].astype(int)
    nodes['lat'] = nodes['lat'].astype(float)
    nodes['lon'] = nodes['lon'].astype(float)

    # format edges
    edges = edges.reset_index()
    edges['source_osmid'] = edges['u']
    edges['target_osmid'] = edges['v']
    edges['source_node'] = edges['source_osmid'].apply(lambda x: nodes.loc[nodes['osmid']==x, 'node_id'].values[0])
    edges['target_node'] = edges['target_osmid'].apply(lambda x: nodes.loc[nodes['osmid']==x, 'node_id'].values[0])
    edges = edges.sort_values(by=['travel_time'])
    edges = edges.drop_duplicates(subset=['source_node', 'target_node'])
    edges = edges[['source_osmid', 'target_osmid', 'source_node', 'target_node', 'travel_time']]
    edges['source_osmid'] = edges['source_osmid'].astype(int)
    edges['target_osmid'] = edges['target_osmid'].astype(int)
    edges['source_node'] = edges['source_node'].astype(int)
    edges['target_node'] = edges['target_node'].astype(int)
    edges['travel_time'] = edges['travel_time'].astype(int)
    # format edge types
    print(f"Number of nodes: {len(nodes)}, number of edges: {len(edges)}")
    return G, nodes, edges


def generateMap(G,nodes,edges):
    OUTPUT_DIR = BASE_OUTPUT_DIR+"map/"

    # shortest_path_length=dict(nx.all_pairs_dijkstra_path_length(G, weight='travel_time'))
    # with open(OUTPUT_DIR+"times.csv", 'a+') as output_file:
    #     for origin in range(len(nodes)):
    #         values = []
    #         origin_osmid = nodes.loc[nodes['node_id']==origin,'osmid'].iloc[0]
    #         for destination in range(len(nodes)):
    #             destination_osmid = nodes.loc[nodes['node_id']==destination,'osmid'].iloc[0]
    #             values.append(shortest_path_length[origin_osmid][destination_osmid])
    #         output_file.write(",".join([str(i) for i in values])+"\n")

    with open(OUTPUT_DIR+"pred.csv", 'a+') as pred_file:
        with open(OUTPUT_DIR+"times.csv", 'a+') as times_file:
            for origin in range(len(nodes)):
                travel_times = []
                predecessors = []
                origin_osmid = nodes.loc[nodes['node_id']==origin,'osmid'].iloc[0]
                pred,travel_time=nx.dijkstra_predecessor_and_distance(G, origin_osmid,weight='travel_time')
                for destination in range(len(nodes)):
                    destination_osmid = nodes.loc[nodes['node_id']==destination,'osmid'].iloc[0]
                    travel_times.append(travel_time[destination_osmid])
                    if destination == origin:
                        predecessor = 0
                    else:
                        predecessor = nodes.loc[nodes['osmid']==pred[destination_osmid][0],'node_id'].iloc[0]+1
                    predecessors.append(predecessor)
                pred_file.write(",".join([str(i) for i in predecessors])+"\n")
                times_file.write(",".join([str(i) for i in travel_times])+"\n")

    nodes['node_id'] = nodes['node_id'].apply(lambda x: x + 1)
    nodes = nodes[['node_id', 'lat', 'lon']]
    nodes.to_csv(OUTPUT_DIR+'nodes.csv', header=False, index=False)

    edges['source_node'] = edges['source_node'].apply(lambda x: x + 1)
    edges['target_node'] = edges['target_node'].apply(lambda x: x + 1)
    edges['travel_time'] = edges['travel_time'].apply(lambda x: math.ceil(x))
    edges = edges[['source_node', 'target_node', 'travel_time']]
    edges.to_csv(OUTPUT_DIR+'edges.csv', header=False, index=False)

def generateRequests(G,nodes,address,filename,start,end):

    DROPOFF_LAT = 'dropoff_latitude'
    DROPOFF_LONG = 'dropoff_longitude'
    PICKUP_LAT = 'pickup_latitude'
    PICKUP_LONG = 'pickup_longitude'
    PICKUP_TIME = 'tpep_pickup_datetime'
    TRIP_DISTANCE = 'trip_distance'
    ORIGIN = 'origin'
    DEST = 'dest'

    # def getData(filename):
    #     return pd.read_csv(filename,parse_dates=True, usecols=[PICKUP_TIME, PICKUP_LONG, PICKUP_LAT, DROPOFF_LONG, DROPOFF_LAT, TRIP_DISTANCE]).sort_values(by = [PICKUP_TIME])

    city = ox.geocode_to_gdf(address)
    geom = city.loc[0, 'geometry']

    data = pd.read_csv(filename,parse_dates=True, usecols=[PICKUP_TIME, PICKUP_LONG, PICKUP_LAT, DROPOFF_LONG, DROPOFF_LAT, TRIP_DISTANCE]).sort_values(by = [PICKUP_TIME])
    data = data.loc[(data[PICKUP_TIME] >= start) & 
                                (data[PICKUP_TIME] < end)]
    count = data.shape[0]
    data.insert(6,ORIGIN,np.zeros(count))
    data.insert(7,DEST,np.zeros(count))

    print("Trips in selected time interval: {0}".format(count))

    selected_trips = []
    num_processed = 0
    for index, row in data.iterrows():
            print(num_processed, end='\r')
            num_processed+=1
            cord1 = (row[PICKUP_LONG],row[PICKUP_LAT])
            cord2 = (row[DROPOFF_LONG], row[DROPOFF_LAT])
            fit1 = geom.intersects(Point(cord1))
            fit2 = geom.intersects(Point(cord2))
            if fit1 and fit2:
                ori = ox.distance.nearest_nodes(G, row[PICKUP_LONG],row[PICKUP_LAT])
                des = ox.distance.nearest_nodes(G, row[DROPOFF_LONG], row[DROPOFF_LAT])
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
                    
                data.at[index,ORIGIN] = nodes.loc[nodes['osmid']==origin,'node_id'].iloc[0]+1
                data.at[index,DEST] = nodes.loc[nodes['osmid']==destination,'node_id'].iloc[0]+1
                # if origin != destination and (origin in shortest_path_length and destination in shortest_path_length[origin]):
                selected_trips.append(index)

    print("Trips after filtering: {0}".format(len(selected_trips)))
    data.loc[selected_trips].to_csv(BASE_OUTPUT_DIR+"requests/requests.csv")
#  getData('yellow_tripdata_2015-01.csv')


def generateVehicles(nodes,vehicle_num,vehicle_capacity):
    vehicles = pd.DataFrame()
    # randomly generate starting points 
    start_node = random.sample(nodes.index.to_list(), vehicle_num)
    for n in start_node:
        vehicles = vehicles.append(nodes.iloc[n], ignore_index=True)

    vehicles.columns = ['node id', 'node lat', 'node lon']
    vehicles['vehicle id'] = list(range(1, vehicle_num+1))
    vehicles['start time'] = datetime.time(0,0,0)
    vehicles['capacity'] = vehicle_capacity

    # formatting
    vehicles['node id'] = vehicles['node id'].astype('int')
    vehicles = vehicles[['vehicle id', 'node id', 'node lat', 'node lon', 'start time', 'capacity']]
    vehicles.to_csv(BASE_OUTPUT_DIR+'/vehicles/vehicles.csv', index = False, header = False)

G, nodes, edges = osmnx_routing_graph(addr='Manhattan, New York City, New York, USA', 
                        buffer_dist=500)
# ox.plot_graph(G, node_color='b', node_size=2, node_edgecolor='grey', bgcolor = 'white')

generateRequests(G,nodes,'Manhattan, New York City, New York, USA','yellow_tripdata_2015-01.csv',"2015-01-01 00:00:00","2015-01-02 00:00:00")

generateMap(G,nodes,edges)

generateVehicles(nodes,10000,4)