import networkx as nx
import pickle
import pandas as pd
import numpy as np
import math
from dateutil import parser
import datetime
import mosek
import time 

DROPOFF_LAT = 'dropoff_latitude'
DROPOFF_LONG = 'dropoff_longitude'
PICKUP_LAT = 'pickup_latitude'
PICKUP_LONG = 'pickup_longitude'
PICKUP_TIME = 'tpep_pickup_datetime'
TRIP_DISTANCE = 'trip_distance'
ORIGIN = 'origin'
DEST = 'dest'

Manhattan_network = pickle.load(open("manhatton/Manhattan_network.p", "rb"))
shortest_path_length=dict(nx.all_pairs_dijkstra_path_length(Manhattan_network, weight='length'))
shortest_paths = nx.shortest_path(Manhattan_network)
selected = pd.read_csv("manhatton/input.csv",parse_dates=True).sort_values(by = [PICKUP_TIME])

TRIP_ID = "trip_id"
ARRIVAL_TIME = "arrival_time"
DEPARTURE_TIME = "departure_time"
STOP_ID = "stop_id"
STOP_SEQUENCE = "stop_sequence"

STOP_ID = "stop_id"
STOP_LAT = "stop_lat"
STOP_LON = "stop_lon"

RESULT_DIRECTORY = "results/withbus/"

def getBusRoute(trip_id):
    return trip_id.split("_")[-2]

frequency = {"M1":3,"M2":4,"M3":4,"M5":4,"M4":4,"M7":4,"M8":6,
             "M9":6,"M10":3,"M11":4,"M12":2,"M20":2,"M21":2,"M31":4,
             "M42":6,"M50":3,"M55":3,"M57":6,"M66":3,"M72":3,"M96":12,"M98":3,"M101":6,"M104":6}
times = pd.read_csv("manhatton/trip_times.txt")
bus_routes_meta = {}
prev_trip_id = None
way = 0
for index, row in times.iterrows():
    trip_id = row[TRIP_ID]
    route_id = getBusRoute(trip_id)
    if prev_trip_id != trip_id:
        if route_id not in bus_routes_meta:
            bus_routes_meta[route_id] = {}
            bus_routes_meta[route_id][0] = []
            bus_routes_meta[route_id][1] = []
            way = 0
        else:
            way = 1
    bus_routes_meta[route_id][way].append(row[STOP_ID])
    prev_trip_id = trip_id


WAITING_TIME = "wait"
FREQUENCY = "frequency"
STOPS = "stops"
ROUTE_LENGTH = "route_length"
TRAVEL_BETWEEN_STOPS = "travel_between_stops"
# FIRST_BUS = "first_bus"
ID="id"
bus_speed = 10

def getNoOfBussesWaitTime(route_length,frequency,speed):
    travel_time = route_length/(speed*1000)
    no_of_busses =  math.ceil(travel_time*frequency)
    waiting_time = (no_of_busses/frequency)-travel_time
    return waiting_time

def getStopDetails(line,shortest_path_lengths,stops,speed,frequency):
    distances = [0]
    first_bus_in_hour = [0]
    distanceFromStart = 0
    for i in range(1,len(stops)):
        distanceFromLastStop = shortest_path_lengths[stops[i-1]][stops[i]]
        distanceFromStart += distanceFromLastStop
        
        distances.append(distanceFromStart)
        time_to_stop = distanceFromStart/(speed*1000)
        frac,_ = math.modf(time_to_stop)
        first_bus_in_hour.append(frac%(1/frequency))
    
    distances_between_stops = {}
    for i,stop1 in enumerate(stops[:-1]):
        if stop1 not in distances_between_stops:
            distances_between_stops[stop1] = {}
        for j,stop2 in enumerate(stops):
                if j!=0 and stop1 != stop2:
#                     if line == "M2":
#                         if stop1==42436578 and stop2== 42430872:
#                             print("fdjvndfjv")
                    if i < j:
                        distance_between = distances[j]-distances[i]
                        if stop2 not in distances_between_stops[stop1]:
                            distances_between_stops[stop1][stop2] = (distance_between,0,first_bus_in_hour[i])
                        if distance_between < distances_between_stops[stop1][stop2][0]:
                            distances_between_stops[stop1][stop2] = (distance_between,0,first_bus_in_hour[i])
                    else:
                        distance_between = distances[j]+distanceFromStart-distances[i]
                        if stop2 not in distances_between_stops[stop1]:
                            distances_between_stops[stop1][stop2] = (distance_between,1,first_bus_in_hour[i])
                        if distance_between < distances_between_stops[stop1][stop2][0]:
                            distances_between_stops[stop1][stop2] = (distance_between,1,first_bus_in_hour[i])
    return distances,distances_between_stops

def getRouteLength(shortest_path_lengths,route):
    routeLength = 0
    for i in range(1,len(route)):
        routeLength += shortest_path_lengths[route[i-1]][route[i]]
    return routeLength

def getBusLines(G,bus_routes_meta,shortest_paths,shortest_path_lengths,bus_speed,frequencies):
    bus_lines = {}
    for route_key in bus_routes_meta:
        route = bus_routes_meta[route_key]
        stops = route[0]+route[1]
        freq = frequencies[route_key]
        distances,distances_between_stops = getStopDetails(route_key,shortest_path_lengths,stops,bus_speed,freq)
        route_length = getRouteLength(shortest_path_lengths,stops)
        waiting_time = getNoOfBussesWaitTime(route_length,freq,bus_speed)
        print("{0},{1},{2}".format(route_key,route_length,route_length/(bus_speed*1000)))
        bus_line = {WAITING_TIME:waiting_time,FREQUENCY:freq,STOPS:stops,TRAVEL_BETWEEN_STOPS:distances_between_stops}
        bus_lines[route_key] = bus_line
    return bus_lines

def getServableRoutes(G,bus_lines,shortest_path_lengths,cutoff):
    servableRoutes = {}
    closestToStop = {}
    closestFromStop = {}
    key_errors = 0
    for node in G.nodes:
        servableRoutes[node] = []
        closestToStop[node] = {}
        closestFromStop[node] = {}
    for line_id in bus_lines:
        stops = bus_lines[line_id][STOPS]
        for stop in stops:
            for node in G.nodes:
                try:
                    distance = min(shortest_path_lengths[node][stop],shortest_path_lengths[stop][node])
                    if distance <= cutoff:
                        if line_id not in servableRoutes[node]:
                            servableRoutes[node].append(line_id)
                            closestToStop[node][line_id] = stop
                            closestFromStop[node][line_id] = stop
                        current_close_stop = closestToStop[node][line_id]
                        current_distance = shortest_path_lengths[node][current_close_stop]
                        if shortest_path_lengths[node][stop] < current_distance:
                            closestToStop[node][line_id] = stop
                        current_from_stop = closestFromStop[node][line_id]
                        current_distance = shortest_path_lengths[current_from_stop][node]
                        if shortest_path_lengths[stop][node] < current_distance:
                            closestFromStop[node][line_id] = stop
                except KeyError:
                    key_errors+=1
    return servableRoutes,closestToStop,closestFromStop


buslines = getBusLines(Manhattan_network,bus_routes_meta,shortest_paths,shortest_path_length,bus_speed,frequency)
servable_routes,closest_to_stop,closest_from_stop = getServableRoutes(Manhattan_network,buslines,shortest_path_length,1000)

def getTimeOneBus(line,bus_speed,start,end):
    travel_data = line[TRAVEL_BETWEEN_STOPS][start][end]
    travel_time = travel_data[0]/bus_speed
    if travel_data[1] == 1:
        travel_time += line[WAITING_TIME]
    return travel_time

def getFirstBusArrivalTime(first_bus_in_hour,frequency,start_time):
    minutes_to_first_bus_in_hour = first_bus_in_hour*60
    starting_minute = start_time.minute
    time_gap = 60/frequency
    if minutes_to_first_bus_in_hour < starting_minute:
        return start_time + datetime.timedelta(minutes=minutes_to_first_bus_in_hour - starting_minute + time_gap*math.ceil((starting_minute-minutes_to_first_bus_in_hour)/time_gap))
    if minutes_to_first_bus_in_hour >= starting_minute:
        return start_time + datetime.timedelta(minutes=minutes_to_first_bus_in_hour-starting_minute)

def getBusDetails(line_id,busline,start_node,end_node,start_time,max_travel_time,bus_speed):
    busses = []
    try:
        travel_data = busline[TRAVEL_BETWEEN_STOPS][start_node][end_node]
    except KeyError:
        print(line_id,start_node,end_node)
    travel_time = travel_data[0]/bus_speed
    if travel_data[1] == 1:
        travel_time += busline[WAITING_TIME]
    if travel_time < max_travel_time:
        max_wait_time_for_bus = max_travel_time-travel_time
        firstBusArrivesOn = getFirstBusArrivalTime(travel_data[2],busline[FREQUENCY],start_time)
        waiting_upto = start_time + datetime.timedelta(hours=max_wait_time_for_bus)
        busArriveOn = firstBusArrivesOn
        while busArriveOn <= waiting_upto:
            busses.append(busArriveOn)
            busArriveOn = busArriveOn + datetime.timedelta(hours=(1/busline[FREQUENCY]))
    return busses,travel_time

def getMaximumTripTime(addition_trip_time_factor,shortest_path_lengths,bus_speed,wait_time,origin,dest):
    try:
        maxTime = ((1+addition_trip_time_factor)*shortest_path_lengths[origin][dest])/(bus_speed*1000)+wait_time
    except KeyError:
        print(origin,dest)
        return wait_time
    return maxTime

def generateRBCombinations(G,buslines,shortest_paths,shortest_path_lengths,servable_routes,closest_to_stop,closest_from_stop,bus_speed,taxi_speed,wait_time,addition_trip_time_factor,starting_time,requests):
    combinations = {}
    taxi_speed_in_meters = taxi_speed*1000
    bus_speed_in_meters = bus_speed*1000
    for _, row in requests.iterrows():
        index = row[0]
        combinations[index] = []
        origin = row[ORIGIN]
        dest = row[DEST]
        max_trip_time = getMaximumTripTime(addition_trip_time_factor,shortest_path_lengths,bus_speed,wait_time,origin,dest)
        print(index,max_trip_time)
        bus_lines_close_to_origin = servable_routes[origin]
        bus_lines_close_to_dest = servable_routes[dest]
        taxi_distance = shortest_path_lengths[origin][dest]
        # get single bus trips
        for line in bus_lines_close_to_origin:
            if line in bus_lines_close_to_dest:
                bus_line = buslines[line]
                start = closest_to_stop[origin][line]
                end = closest_from_stop[dest][line]
                if start != end and shortest_path_lengths[origin][start] + shortest_path_lengths[end][dest] < taxi_distance:
                    time_to_start_bus_stop = shortest_path_lengths[origin][start]/taxi_speed_in_meters
                    time_from_bus_stop_to_dest = shortest_path_lengths[end][dest]/taxi_speed_in_meters
                    time_at_start = starting_time+datetime.timedelta(hours=time_to_start_bus_stop)
                    max_bus_travel_time = max_trip_time - time_to_start_bus_stop - time_from_bus_stop_to_dest
#                     if index == 128034 and line == 61:
#                         print(time_at_start,max_bus_travel_time)
                    busses,travel_time = getBusDetails(line,bus_line,start,end,time_at_start,max_bus_travel_time,bus_speed_in_meters)
                    for bus in busses:
                        combinations[index].append((0,line,bus,start,end,bus+datetime.timedelta(hours=travel_time)))
        # get transfer bus trips
        for line1 in bus_lines_close_to_origin:
            for line2 in bus_lines_close_to_dest:
                if line1 != line2:
                    start = closest_to_stop[origin][line1]
                    end = closest_from_stop[dest][line2]
                    if start != end and shortest_path_lengths[origin][start] + shortest_path_lengths[end][dest] < taxi_distance:
                        transfer_node = None
                        distance = -1
                        for stop in buslines[line1][STOPS]:
                            if (stop != start and stop != end) and stop in buslines[line2][STOPS]:
                                try:
#                                     print(line1,line2,start,stop,end)
                                    dist = buslines[line1][TRAVEL_BETWEEN_STOPS][start][stop][0] + buslines[line2][TRAVEL_BETWEEN_STOPS][stop][end][0]
                                    if distance == -1 or dist < distance:
                                        distance = dist
                                        transfer_node = stop
                                except KeyError:
                                    print(start,end,stop,line1,line2)
                        if transfer_node != None:
                            time_to_start_bus_stop = shortest_path_lengths[origin][start]/taxi_speed_in_meters
                            time_from_bus_stop_to_dest = shortest_path_lengths[end][dest]/taxi_speed_in_meters
                            time_at_start = starting_time+datetime.timedelta(hours=time_to_start_bus_stop)
                            max_bus_travel_time = max_trip_time - time_to_start_bus_stop - time_from_bus_stop_to_dest
                            bus_line1 = buslines[line1]
                            bus_line2 = buslines[line2]
                            time_on_bus1 = getTimeOneBus(bus_line1,bus_speed_in_meters,start,transfer_node)
                            time_on_bus2 = getTimeOneBus(bus_line2,bus_speed_in_meters,transfer_node,end)
                            max_bus_travel_time_on_bus1 = max_bus_travel_time - time_on_bus2
                            max_bus_travel_time_on_bus2 = max_bus_travel_time - time_on_bus1
                            busses1,_ = getBusDetails(line1,bus_line1,start,transfer_node,time_at_start,max_bus_travel_time_on_bus1,bus_speed_in_meters)
                            earliest_time_at_bus_line2 = time_at_start + datetime.timedelta(hours=time_on_bus1)
                            busses2,_ = getBusDetails(line2,bus_line2,transfer_node,end,earliest_time_at_bus_line2,max_bus_travel_time_on_bus2,bus_speed_in_meters)
                            for bus1 in busses1:
                                for bus2 in busses2:
                                    bus1_at_transfer_node = bus1 + datetime.timedelta(hours=time_on_bus1)
                                    if bus1_at_transfer_node <= bus2:
                                        bus2_at_end_node = bus2 + datetime.timedelta(hours=time_on_bus2)
                                        if bus2_at_end_node <= starting_time + datetime.timedelta(hours=max_trip_time):
                                            combinations[index].append((1,line1,line2,bus1,bus2,start,transfer_node,end,bus2_at_end_node))  
    return combinations

# combinations = generateRBCombinations(Manhattan_network,buslines,shortest_paths,shortest_path_length,servable_routes,closest_to_stop,closest_from_stop,10,20,1/12,0.25,start_at,sample)


# (#tripNo,#origin/dest,#stop,#reaching_time,#leaving_time)
# (#tripNo,#origin,#dest,earliest_pick_time,#latest_drop_time)

R_TRIP_NO = 0
R_OD = 1
R_STOP = 2
R_REACH = 3
R_LEAVE = 4

T_TRIP_NO = 0
T_ORIGIN = 1
T_DEST = 2
T_PT = 3
T_DT = 4

V_TRIPS = 2
V_LAST_TIME = 1
V_LAST_POS = 0
V_ROUTE = 3
V_NO = 4

def orderByGivenOrder(X,Z):
    return [X[i] for i in Z]

def createVehicleFleat(G,size,starting_time):
    np.random.seed(1)
    nodes = list(G.nodes)
    no_of_nodes = len(nodes)
    fleet = []
    for i in range(size):
        node = np.random.randint(0,no_of_nodes-1)
        fleet.append({V_LAST_POS: nodes[node],V_LAST_TIME: starting_time,V_TRIPS: [],V_ROUTE:[],V_NO: i})
    return fleet

def getRoute(shortest_path_lengths,taxi_speed_in_meters,current_pos,current_time,stops,reach_by,leave_at):
    arrival_times = []
    leaving_times = []
    total_distance = 0
    prev_stop = current_pos
    prev_left_time = current_time
    for index in range(len(stops)):
        stop = stops[index]
        if prev_stop == stop:
            distance = 0
        else:
            try:
                distance = shortest_path_lengths[prev_stop][stop]
            except KeyError:
                return 0,[],[],False
        total_distance += distance
        time_to_travel = distance/taxi_speed_in_meters
        arrived_on = prev_left_time + datetime.timedelta(hours=time_to_travel)
        prev_left_time = arrived_on
        prev_stop = stop
        if reach_by[index] != None and arrived_on > reach_by[index]:
            return 0,[],[],False
        if leave_at[index] != None and arrived_on < leave_at[index]:
            prev_left_time = leave_at[index]
        arrival_times.append(arrived_on)
        leaving_times.append(prev_left_time)
    return total_distance,arrival_times,leaving_times,True

def getVehicleRoutes(combination,stops,arrival_times,leave_at):
    routes = []
    for i in range(len(combination)):
        routes.append((combination[i][0],combination[i][1],stops[i],arrival_times[i],leave_at[i]))
    return routes

def getFinalTripEndTime(combination,arrival_times):
    return arrival_times[combination.index(len(combination)-1)]

def getTripPickUpTime(combination,arrival_times):
    return arrival_times[combination.index(len(combination)-2)]

def updateVehicleWithNewTrip(vehicle,shortest_paths,shortest_path_lengths,taxi_speed,current_time,trip_no,start_node,end_node,earliest_start_time,latest_end_time,add):
    taxi_speed_in_meters = taxi_speed*1000
    current_trips = vehicle[V_TRIPS]
    current_stops = vehicle[V_ROUTE]
    last_updated_time = vehicle[V_LAST_TIME]
    last_position = vehicle[V_LAST_POS]
    last_reached_stop = None
    completed_trips = []
    picked_trips = []
#     print(current_stops)
    for i in range(len(current_stops)):
        stop = current_stops[i]
        if stop[R_REACH] > current_time:
            break
        last_reached_stop = i
        if stop[R_LEAVE] > current_time:
            break
        if stop[R_OD] == 1:
            completed_trips.append(stop[R_TRIP_NO])
            if stop[R_TRIP_NO] in picked_trips:
                picked_trips.remove(stop[R_TRIP_NO])
        else:
            picked_trips.append(stop[R_TRIP_NO])
    for i in completed_trips:
        for trip_index in range(len(vehicle[V_TRIPS])):
            if vehicle[V_TRIPS][trip_index][T_TRIP_NO] == i:
                vehicle[V_TRIPS].pop(trip_index)
                break
    if last_reached_stop != None:
        last_updated_time = vehicle[V_ROUTE][last_reached_stop][R_REACH]
        last_position = vehicle[V_ROUTE][last_reached_stop][R_STOP]
        if vehicle[V_ROUTE][last_reached_stop][R_LEAVE] <= current_time:
            vehicle[V_ROUTE] = vehicle[V_ROUTE][(last_reached_stop+1):]
        else:
            vehicle[V_ROUTE] = vehicle[V_ROUTE][last_reached_stop:]
        vehicle[V_LAST_TIME] = last_updated_time
        vehicle[V_LAST_POS] = last_position
    
    if len(vehicle[V_TRIPS]) > 0 and len(vehicle[V_TRIPS]) < 2:
        next_dest = vehicle[V_ROUTE][0][R_STOP]
        if last_position != next_dest:
            route = shortest_paths[last_position][next_dest]
            distance = 0
            prev_node = last_position
            for node in route:
                distance += shortest_path_lengths[prev_node][node]
                time_to_travel = distance/taxi_speed_in_meters
                time_at_node = last_updated_time + datetime.timedelta(hours=time_to_travel)
                prev_node = node
                if time_at_node >= current_time:
                    last_updated_time = time_at_node
                    last_position = node
                    break
        
        distance_remaining = 0
        prev_stop = last_position
        for route in vehicle[V_ROUTE]:
            if prev_stop != route[R_STOP]:
                distance_remaining += shortest_path_lengths[prev_stop][route[R_STOP]]
            prev_stop = route[R_STOP]
        if len(vehicle[V_TRIPS]) == 1:
            trip1_no = vehicle[V_TRIPS][0][T_TRIP_NO]
            trip1_origin = vehicle[V_TRIPS][0][T_ORIGIN]
            trip1_dest = vehicle[V_TRIPS][0][T_DEST]
            trip1_pt = vehicle[V_TRIPS][0][T_PT]
            trip1_dt = vehicle[V_TRIPS][0][T_DT]
            selected_combination = None
            selected_distance = None
            selected_arrival_times = None
            selected_leave_at = None
            selected_stops = None
            selected_order = None
            order,stops,reach_by,leave_at,combinations = None,None,None,None,None
            combinations = None
            if trip1_no in picked_trips:
                # t_1,s_2,t_2
                order = [(trip1_no,1),(trip_no,0),(trip_no,1)]
                stops,reach_by,leave_at = [trip1_dest,start_node,end_node],[trip1_dt,None,latest_end_time],[None,earliest_start_time,None]
                # (t_1,s_2,t_2),(s_2,t_1,t_2),(t_1,s_2,t_2,t_1)
                combinations = [[0,1,2],[1,0,2],[1,2,0]]
            else:
                # s_1,t_1,s_2,t_2
                order = [(trip1_no,0),(trip1_no,1),(trip_no,0),(trip_no,1)]
                stops,reach_by,leave_at = [trip1_origin,trip1_dest,start_node,end_node],[None,trip1_dt,None,latest_end_time],[trip1_pt,None,earliest_start_time,None]
                # (s_1,t_1,s_2,t_2),(s_1,s_2,t_1,t_2),(s_1,s_2,t_2,t_1),(s_2,t_2,s_1,t_1),(s_2,s_1,t_2,t_1),(s_2,s_1,t_1,t_2)
                combinations = [[0,1,2,3],[0,2,1,3],[0,2,3,1],[2,3,0,1],[2,0,3,1],[2,0,1,3]]
            for combination in combinations:
                c_order = orderByGivenOrder(order,combination)
                c_stops = orderByGivenOrder(stops,combination)
                c_reach_by = orderByGivenOrder(reach_by,combination)
                c_leave_at = orderByGivenOrder(leave_at,combination)
                distance,arrival_times,leaving_times,feasible = getRoute(shortest_path_lengths,taxi_speed_in_meters,last_position,last_updated_time,c_stops,c_reach_by,c_leave_at)
                if feasible:
                    if selected_combination == None or selected_distance > distance:
                        selected_combination = combination
                        selected_distance = distance
                        selected_arrival_times = arrival_times
                        selected_leave_at = leaving_times
                        selected_stops = c_stops
                        selected_order = c_order
            if selected_combination != None:
                if add:
                    vehicle[V_LAST_TIME] = last_updated_time
                    vehicle[V_LAST_POS] = last_position
                    vehicle[V_TRIPS].append((trip_no,start_node,end_node,earliest_start_time,latest_end_time))
#                     print(selected_order,selected_stops,selected_arrival_times,selected_leave_at)
                    vehicle[V_ROUTE] = getVehicleRoutes(selected_order,selected_stops,selected_arrival_times,selected_leave_at)
                return vehicle,True,selected_distance-distance_remaining,getFinalTripEndTime(selected_combination,selected_arrival_times),getTripPickUpTime(selected_combination,selected_arrival_times)
    elif len(vehicle[V_ROUTE]) == 0:
        order = [(trip_no,0),(trip_no,1)]
        stops,reach_by,leave_at = [start_node,end_node],[None,latest_end_time],[earliest_start_time,None]
        distance,arrival_times,leaving_times,feasible = getRoute(shortest_path_lengths,taxi_speed_in_meters,last_position,current_time,stops,reach_by,leave_at)
#         print(distance,arrival_times,feasible)
        if feasible:
            vehicle[V_LAST_TIME] = current_time
            vehicle[V_LAST_POS] = last_position
            if add:
                vehicle[V_TRIPS].append((trip_no,start_node,end_node,earliest_start_time,latest_end_time))
                vehicle[V_ROUTE] = getVehicleRoutes(order,stops,arrival_times,leaving_times)
            return vehicle,True,distance,getFinalTripEndTime([0,1],arrival_times),getTripPickUpTime([0,1],arrival_times)
        
#         time_at_pick_up = current_time + datetime.timedelta(hours=(shortest_path_lengths[last_position][start_node]/taxi_speed_in_meters))
#         leave_at_pick_up = time_at_pick_up
#         if time_at_pick_up < earliest_start_time:
#             leave_at_pick_up = earliest_start_time
#         time_at_dest = leave_at_pick_up + datetime.timedelta(hours=(shortest_path_lengths[start_node][end_node]/taxi_speed_in_meters))
#         if time_at_dest <= latest_end_time:
#             if add:
#                 vehicle[V_TRIPS].append((trip_no,start_node,end_node,earliest_start_time,latest_end_time))
#                 vehicle[V_ROUTE].append((trip_no,0,start_node,time_at_pick_up,earliest_start_time))
#                 vehicle[V_ROUTE].append((trip_no,1,end_node,time_at_dest))
#             return vehicle,True,shortest_path_lengths[start_node][end_node]+shortest_path_lengths[last_position][start_node]
    return vehicle,False,0,None,None

# system_start_time = parser.parse("2015-01-01 00:00:00")
# vehicles = createVehicleFleat(Manhattan_network,1000,system_start_time)
# VRB,RBV,VR,no_combns,no_vars = getVRBCombinations(sample,combinations,vehicles,0.25,1/12,shortest_paths,shortest_path_length,bus_speed,taxi_speed,start_at)
# vehicle = {0: 42443332, 
#            1: datetime.datetime(2015, 1, 1, 11, 44, 34), 
#            2: [(128086, 42443329.0, 1918039904.0, datetime.datetime(2015, 1, 1, 11, 44, 34), datetime.datetime(2015, 1, 1, 12, 38, 46, 244570))], 
#            3: [(128086, 0, 42443329.0, datetime.datetime(2015, 1, 1, 11, 44, 48, 278636), datetime.datetime(2015, 1, 1, 11, 44, 48, 278636)), (128086, 1, 1918039904.0, datetime.datetime(2015, 1, 1, 12, 4, 29, 176464), datetime.datetime(2015, 1, 1, 12, 4, 29, 176464))], 
#            4: 13}
# current_time = parser.parse("2015-01-01 12:5:49")
# trip_no = 128089
# start_node = 42436705
# end_node = 42440452 
# earliest_start_time = parser.parse("2015-01-01 11:44:49")
# latest_end_time = parser.parse("2015-01-01 12:46:47.235610")
# updateVehicleWithNewTrip(vehicle,shortest_paths,shortest_path_length,taxi_speed,current_time,trip_no,start_node,end_node,earliest_start_time,latest_end_time,True)

#(#B/BB, #R_no, (#R_no2), #Pick_time, (#transfer_time), #start, (#transfer_node), #end, #drop_time)
taxi_speed = 20

def getVRBCombinations(trips,RBCombinations,vehicles,addition_trip_time_factor,wait_time,shortest_paths,shortest_path_lengths,bus_speed,taxi_speed,current_time):
    VRB = {}
    VRB_indices = {}
    RBV = {}
    RBV_indices = {}
    VR = {}
    no_combns = 0
    no_vars = 0
    for R in RBCombinations:
        trip = trips.loc[R]
        origin_node = trip[ORIGIN]
        destination_node = trip[DEST]
        starting_time = parser.parse(trip[PICKUP_TIME])
        RBs = RBCombinations[R]
        max_trip_time = getMaximumTripTime(addition_trip_time_factor,shortest_path_lengths,bus_speed,wait_time,origin_node,destination_node)
        final_latest_time = starting_time + datetime.timedelta(hours=max_trip_time)
        trip_no,start_node,end_node,earliest_start_time,latest_end_time = R,origin_node,None,current_time,None
        start_node_lm,earliest_start_time_lm = None,None
        RB_no = 0
        VRB[trip_no] = []
        RBV[trip_no] = []
        VRB_indices[trip_no] = []
        RBV_indices[trip_no] = []
        for RB in RBs:
#             VRB[trip_no].append([])
#             RBV[trip_no].append([])
            first_mile = []
            last_mile = []
            if RB[0] == 0:
                end_node,latest_end_time = RB[3],RB[2]
            else:
                end_node,latest_end_time = RB[5],RB[3]
            if end_node == origin_node or shortest_path_lengths[origin_node][end_node]<=200:
                first_mile.append((RB_no,-1))
            else:
                for vehicle in vehicles:
                    _,feasible,distance,_,_ = updateVehicleWithNewTrip(vehicle,shortest_paths,shortest_path_lengths,taxi_speed,current_time,trip_no,start_node,end_node,earliest_start_time,latest_end_time,False)
                    if feasible:
                        first_mile.append((RB_no,vehicle[V_NO],distance))
                    
            if RB[0] == 0:
                start_node_lm,earliest_start_time_lm = RB[4],RB[5]
            else:
                start_node_lm,earliest_start_time_lm = RB[7],RB[8]
            if start_node_lm == destination_node or shortest_path_lengths[start_node_lm][destination_node]<=200:
                last_mile.append((RB_no,-1,(earliest_start_time_lm-current_time).seconds))
            else:
                for vehicle in vehicles:
                    _,feasible,distance,finishing_time,_ = updateVehicleWithNewTrip(vehicle,shortest_paths,shortest_path_lengths,taxi_speed,current_time,trip_no,start_node_lm,destination_node,earliest_start_time_lm,final_latest_time,False)
                    if feasible:
                        last_mile.append((RB_no,vehicle[V_NO],distance,(finishing_time-current_time).seconds))
            if len(last_mile)>0 and len(first_mile)>0:
                VRB[trip_no].append([])
                RBV[trip_no].append([])
                for i in first_mile:
                    VRB[trip_no][-1].append(i)
                for i in last_mile:
                    RBV[trip_no][-1].append(i)
                no_combns +=1
                no_vars+=len(last_mile)+len(first_mile)
            RB_no+=1
    i = 0
    for index, row in trips.iterrows():
        no_veh_support = 0
        origin = row[ORIGIN]
        dest = row[DEST]
        trip_no = index
#         VR[trip_no] = []
        starting_time = parser.parse(row[PICKUP_TIME])
        max_trip_time = getMaximumTripTime(addition_trip_time_factor,shortest_path_lengths,bus_speed,wait_time,origin,dest)
        final_latest_time = starting_time + datetime.timedelta(hours=max_trip_time)
        for vehicle in vehicles:
            _,feasible,distance,finishing_time,_ = updateVehicleWithNewTrip(vehicle,shortest_paths,shortest_path_lengths,taxi_speed,current_time,trip_no,origin,dest,current_time,final_latest_time,False)
            if feasible:
                if no_veh_support == 0:
                    VR[trip_no] = []
                VR[trip_no].append((vehicle[V_NO],distance,(finishing_time-current_time).seconds))
                no_vars+=1
                no_veh_support+=1
    return VRB,RBV,VR,no_combns,no_vars


def solveTheProblem(vehicles,trips,VRBs,RBVs,VRs,no_vars,no_comb,penalty,ipmSolverTimeOut):
    trip_count = trips.shape[0]
    no_of_veh = len(vehicles)
    numvar = no_vars+trip_count
    x = np.zeros(numvar)
    c = np.zeros(numvar)
    numcon = 2*trip_count+no_comb+no_of_veh
#     A = np.zeros((numcon,numvar))
    trip_index = {}
    i = 0
    for index, row in trips.iterrows():
        trip_index[index] = i
        i+=1
    k = numcon-no_of_veh
    j = 0
    rb_no = 2*trip_count
    
    with mosek.Env() as env:                            # Create Environment
        with env.Task(0, 1) as task:
            task.appendcons(numcon)
            task.appendvars(numvar)
    
            # Select one from 
            for trip_no in VRBs:
                RBs = VRBs[trip_no]
                i = trip_index[trip_no]
                if len(RBs) > 0:
                    for rb_index in range(len(RBs)):
                        vrbs = RBs[rb_index]
#                         print(len(vrbs))
                        for vrb in vrbs:
#                             A[i,j]=1
#                             A[rb_no,j]=1
                            asub = [i,rb_no]
                            vals = [1,1]
                            if vrb[1] != -1:
                                task.putcj(j, vrb[2])
                                asub.append(k+vrb[1])
                                vals.append(1)
#                                 A[k+vrb[1],j]=1
                            else:
                                task.putcj(j, 0)
                            task.putvarbound(j, mosek.boundkey.ra, 0, 1)
#                             print(asub,vals)
                            task.putacol(j,asub,vals)
                            j+=1
                        rb_no+=1

            rb_no = 2*trip_count
            for trip_no in VRBs:
                RBs = VRBs[trip_no]
                i = trip_index[trip_no]
                if len(RBs) > 0:
                    for rb_index in range(len(RBs)):
                        rbvs = RBVs[trip_no][rb_index]
                        for rbv in rbvs:
#                             A[i+trip_count,j]=1
#                             A[rb_no,j]=-1
                            asub = [i+trip_count,rb_no]
                            vals = [1,-1]
                            if rbv[1] != -1:
                                task.putcj(j, rbv[2])
                                asub.append(k+rbv[1])
                                vals.append(1)
#                                 A[k+rbv[1],j]=1
                            else:
                                task.putcj(j, 0)
                            task.putvarbound(j, mosek.boundkey.ra, 0, 1)
#                             print(j,asub,vals,i,trip_count)
                            task.putacol(j,asub,vals)
                            j+=1
                        rb_no+=1
            
            for trip_no in VRs:
                R = VRs[trip_no]
                i = trip_index[trip_no]
                for vr in R:
#                     A[i,j]=1
#                     A[i+trip_count,j]=1
#                     A[k+vr[0],j]=1
                    asub = [i,i+trip_count,k+vr[0]]
                    vals = [1,1,1]
#                     print(j)
                    task.putcj(j, vr[1])
                    task.putvarbound(j, mosek.boundkey.ra, 0, 1)
                    task.putacol(j,asub,vals)
                    j+=1
                rb_no+=1
            for i in range(trip_count):
#                 A[i,j]=1
#                 A[i+trip_count,j]=1
                asub = [i,i+trip_count]
                vals = [1,1]
                task.putcj(j, penalty)
                task.putvarbound(j, mosek.boundkey.ra, 0, 1)
                task.putacol(j,asub,vals)
                j+=1
                
            task.putconboundlist(range(2*trip_count), [mosek.boundkey.fx]*(2*trip_count), [1]*(2*trip_count), [1]*(2*trip_count))
            z = numcon-no_of_veh-2*trip_count
            task.putconboundlist(range(2*trip_count,numcon-no_of_veh), [mosek.boundkey.fx]*z, [0]*z, [0]*z)
            task.putconboundlist(range(numcon-no_of_veh,numcon), [mosek.boundkey.up]*no_of_veh, [0]*no_of_veh, [1]*no_of_veh)
            
            task.putobjsense(mosek.objsense.minimize)
            task.putvartypelist(np.arange(numvar),
                        [mosek.variabletype.type_int]*numvar)

    
            task.putdouparam(mosek.dparam.mio_max_time, ipmSolverTimeOut)

#             st = time.time()
            task.optimize()                      # Optimize

            task.getxx(mosek.soltype.itg, x)
            
            prosta = task.getprosta(mosek.soltype.itg)
            solsta = task.getsolsta(mosek.soltype.itg)
            if solsta in [mosek.solsta.integer_optimal]:
                print("Optimal solution")
            elif solsta == mosek.solsta.prim_feas:
                print("Feasible solution")
            elif mosek.solsta.unknown:
                if prosta == mosek.prosta.prim_infeas_or_unbounded:
                    print("Problem status Infeasible or unbounded.\n")
                elif prosta == mosek.prosta.prim_infeas:
                    print("Problem status Infeasible.\n")
                elif prosta == mosek.prosta.unkown:
                    print("Problem status unkown.\n")
                else:
                    print("Other problem status.\n")
            else:
                print("Other solution status")
#             print(task.getprimalobj(mosek.soltype.bas))
#             for i in range(c.shape[0]):
#                 c[i] = task.getcj(i)
#             return x,c,A
            return x


A_TYPE = "type"
A_BUSSES = "busses"
A_FTAXI = "ftaxi"
A_LTAXI = "ltaxi"
A_TAXI = "taxi"

A_T_no = "no"
A_T_pick = "pick"
A_T_drop = "drop"

B_T_no = "no"
B_T_start_t = "pickt"
B_T_end_t = "dropt"
B_T_start_n = "pickn"
B_T_end_n = "dropn"

S_taxi_miles = "taxi_miles"
S_taxi_only_trips = "taxi_only_trips"
S_one_bus_trips = "one_bus_trips"
S_two_bus_trips = "two_bus_trips"
S_unserved_trips = "unserved_trips"
S_starts_at = "starts_at"

#(#B/BB, #R_no, (#R_no2), #Pick_time, (#transfer_time), #start, (#transfer_node), #end, #drop_time)
def getBussesInTrip(RBcombination):
    busses = []
    atype = 0
    if RBcombination[0] == 0:
        busses.append({B_T_no:RBcombination[1],B_T_start_t:RBcombination[2],B_T_end_t:RBcombination[5],B_T_start_n:RBcombination[3],B_T_start_n:RBcombination[4]})
    else:
        atype = 1
        busses.append({B_T_no:RBcombination[1],B_T_start_t:RBcombination[3],B_T_end_t:RBcombination[4],B_T_start_n:RBcombination[5],B_T_start_n:RBcombination[6]})
        busses.append({B_T_no:RBcombination[2],B_T_start_t:RBcombination[4],B_T_end_t:RBcombination[8],B_T_start_n:RBcombination[6],B_T_start_n:RBcombination[7]})
    return atype,busses

def getFTaxiInTrip(shortest_paths,shortest_path_lengths,taxi_speed,current_time,trip_no,trip,RBcombination,vehicle):
    start_node,end_node,earliest_start_time,latest_end_time = trip[ORIGIN],None,current_time,None
    if RBcombination[0] == 0:
        end_node,latest_end_time = RBcombination[3], RBcombination[2]
    else:
        end_node,latest_end_time = RBcombination[5], RBcombination[3]
    vehicle,_,_,drop,arrival = updateVehicleWithNewTrip(vehicle,shortest_paths,shortest_path_lengths,taxi_speed,current_time,trip_no,start_node,end_node,earliest_start_time,latest_end_time,True)
    return vehicle,{A_T_no: vehicle[V_NO], A_T_pick: arrival, A_T_drop: drop}

def getLTaxiInTrip(addition_trip_time_factor,bus_speed,wait_time,shortest_paths,shortest_path_lengths,taxi_speed,current_time,trip_no,trip,RBcombination,vehicle):
    start_node,end_node,earliest_start_time,latest_end_time = None,trip[DEST],None,None
    if RBcombination[0] == 0:
        start_node, earliest_start_time = RBcombination[4], RBcombination[5]
    else:
        start_node, earliest_start_time = RBcombination[7], RBcombination[8]
    
    max_trip_time = getMaximumTripTime(addition_trip_time_factor,shortest_path_lengths,bus_speed,wait_time,trip[ORIGIN],trip[DEST])
    latest_end_time = parser.parse(trip[PICKUP_TIME]) + datetime.timedelta(hours=max_trip_time)
    
    vehicle,_,_,drop,arrival = updateVehicleWithNewTrip(vehicle,shortest_paths,shortest_path_lengths,taxi_speed,current_time,trip_no,start_node,end_node,earliest_start_time,latest_end_time,True)
    return vehicle,{A_T_no: vehicle[V_NO], A_T_pick: arrival, A_T_drop: drop}

def getTaxiInTrip(addition_trip_time_factor,bus_speed,wait_time,shortest_paths,shortest_path_lengths,taxi_speed,current_time,trip_no,trip,vehicle):
    start_node,end_node,earliest_start_time,latest_end_time = trip[ORIGIN],trip[DEST],current_time,None
    max_trip_time = getMaximumTripTime(addition_trip_time_factor,shortest_path_lengths,bus_speed,wait_time,trip[ORIGIN],trip[DEST])
    latest_end_time = parser.parse(trip[PICKUP_TIME]) + datetime.timedelta(hours=max_trip_time)
    vehicle,_,_,drop,arrival = updateVehicleWithNewTrip(vehicle,shortest_paths,shortest_path_lengths,taxi_speed,current_time,trip_no,start_node,end_node,earliest_start_time,latest_end_time,True)
    return vehicle,{A_T_no: vehicle[V_NO], A_T_pick: arrival, A_T_drop: drop}


def assignTrips(vehicles,shortest_paths,shortest_path_lengths,taxi_speed,addition_trip_time_factor,bus_speed,wait_time,current_time,trips,x,VRBs,RBVs,VRs,RBcombinations):
    # {type: #RB/RBB/V, busses: [], taxi_first: {}, taxi_last: {}, taxi: {}}
    # bus (#B_route_no, #start_node, #end_node, #pickuptime, #getofftime)
    # taxi (#veh_no, #pick_up_time, #dropoff_time)
    stats = {S_starts_at:current_time, S_taxi_miles:0, S_taxi_only_trips:0, S_one_bus_trips:0, S_two_bus_trips:0,S_unserved_trips:0}
    veh_used = []
    trip_assignment = {}
    trip_index = {}
    i = 0
    j = 0
    for index, row in trips.iterrows():
        trip_index[index] = i
        i+=1
        
    for trip_no in VRBs:
        RBs = VRBs[trip_no]
        i = trip_index[trip_no]
        trip = trips.loc[trip_no]
        if len(RBs) > 0:
            for rb_index in range(len(RBs)):
                vrbs = RBs[rb_index]
                for vrb in vrbs:
                    if x[j] == 1:
                        RBcombination = RBcombinations[trip_no][vrb[0]]
                        a_type,busses = getBussesInTrip(RBcombination)
                        if a_type == 0:
                            stats[S_one_bus_trips] = stats[S_one_bus_trips] +1
                        else:
                            stats[S_two_bus_trips] = stats[S_two_bus_trips] +1
#                         print(stats)
                        trip_assignment[i] = {A_TYPE: a_type, A_BUSSES: busses}
                        veh_no = vrb[1]
                        if veh_no != -1:
#                             print(trip_no,vrb,RBcombination)
                            vehicle,taxi = getFTaxiInTrip(shortest_paths,shortest_path_lengths,taxi_speed,current_time,trip_no,trip,RBcombination,vehicles[veh_no])
                            trip_assignment[i][A_FTAXI] = taxi
                            vehicles[veh_no] = vehicle
                            stats[S_taxi_miles] = stats[S_taxi_miles] +vrb[2]
                    j+=1

    for trip_no in VRBs:
        RBs = VRBs[trip_no]
        i = trip_index[trip_no]
        trip = trips.loc[trip_no]
        if len(RBs) > 0:
            for rb_index in range(len(RBs)):
                rbvs = RBV[trip_no][rb_index]
                for rbv in rbvs:
                    if x[j] == 1:
                        if i not in trip_assignment:
                            with open(RESULT_DIRECTORY+"error.txt", 'a') as file:
                                file.write(str(trip_no)+"\n")
                            print("error ",trip_no)
                        else:
                            RBcombination = RBcombinations[trip_no][rbv[0]]
                            veh_no = rbv[1]
                            if veh_no != -1:
    #                             print(trip_no,rbv,RBcombination)
                                vehicle,taxi = getLTaxiInTrip(addition_trip_time_factor,bus_speed,wait_time,shortest_paths,shortest_path_lengths,taxi_speed,current_time,trip_no,trip,RBcombination,vehicles[veh_no])
                                trip_assignment[i][A_LTAXI] = taxi
                                vehicles[veh_no] = vehicle
                                stats[S_taxi_miles] = stats[S_taxi_miles] +rbv[2]
                    j+=1


    for trip_no in VRs:
        R = VRs[trip_no]
        i = trip_index[trip_no]
        trip = trips.loc[trip_no]
        for vr in R:
            if x[j] == 1:
                stats[S_taxi_only_trips] = stats[S_taxi_only_trips] +1
                trip_assignment[i] = {A_TYPE: 2}
                veh_no = vr[0]
#                 print(trip_no,vr,trip_no)
                vehicle,taxi = getTaxiInTrip(addition_trip_time_factor,bus_speed,wait_time,shortest_paths,shortest_path_lengths,taxi_speed,current_time,trip_no,trip,vehicles[veh_no])
                vehicles[veh_no] = vehicle
                trip_assignment[i][A_TAXI] = taxi
                stats[S_taxi_miles] = stats[S_taxi_miles] +vr[1]
            j+=1
    stats[S_unserved_trips] = trips.shape[0]-stats[S_taxi_only_trips]-stats[S_one_bus_trips]-stats[S_two_bus_trips]
    return trip_assignment, stats


system_start_time = parser.parse("2015-01-01 00:00:00")
vehicles = createVehicleFleat(Manhattan_network,10000,system_start_time)
addition_trip_time_factor = 0.25
wait_time = 1/12

def saveAssignment(base_name,index,item):
    with open(base_name+str(index)+'.pickle', 'wb') as handle:
        pickle.dump(item, handle, protocol=pickle.HIGHEST_PROTOCOL)

def formatDatetime(datetime_object):
    return datetime_object.strftime("%Y.%m.%d %H:%M:%S")

def formatStat(stat):
    return formatDatetime(stat[S_starts_at])+","+str(stat[S_taxi_miles])+","+str(stat[S_taxi_only_trips])+","+str(stat[S_one_bus_trips])+","+str(stat[S_two_bus_trips])+","+str(stat[S_unserved_trips])

def writeStat(stat,filename):
    with open(filename, 'a') as file:
        file.write(formatStat(stat)+"\n")

def writeTimeEx(time_dif,filename):
    with open(filename, 'a') as file:
        file.write(time_dif+"\n")


ipmSolverTimeOut = 1200

# stats,assignments = [],[]
exe_start_time=time.time()
iteration = 0
starting_sample = 0
batch_size = 100
total_sample_size = selected.shape[0]
while starting_sample < total_sample_size:
    batch_size = min(batch_size,total_sample_size-starting_sample)
    sample = selected.iloc[starting_sample:starting_sample+batch_size]
    start_at = parser.parse(sample.iloc[batch_size-1]["tpep_pickup_datetime"])
    combinations = generateRBCombinations(Manhattan_network,buslines,shortest_paths,shortest_path_length,servable_routes,closest_to_stop,closest_from_stop,10,20,1/12,0.25,start_at,sample)
    print(combinations)
    break
    # VRB,RBV,VR,no_combns,no_vars = getVRBCombinations(sample,combinations,vehicles,0.25,1/12,shortest_paths,shortest_path_length,bus_speed,taxi_speed,start_at)
    # x = solveTheProblem(vehicles,sample,VRB,RBV,VR,no_vars,no_combns,1000000,ipmSolverTimeOut)
    # trip_assignment,stat = assignTrips(vehicles,shortest_paths,shortest_path_length,taxi_speed,addition_trip_time_factor,bus_speed,wait_time,start_at,sample,x,VRB,RBV,VR,combinations)
    # print(stat)
    # saveAssignment(RESULT_DIRECTORY+"assignment",iteration,trip_assignment)
    # writeStat(stat,RESULT_DIRECTORY+"stats.csv")
    # starting_sample += batch_size
    # iteration+=1
    # writeTimeEx(str(time.time()-exe_start_time),RESULT_DIRECTORY+"times.txt")
    # exe_start_time = time.time()
