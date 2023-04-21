from structure.node import Node
import requests
from datetime import timedelta
import time

class NetworkHandler:
    def init():
        global routing_url,nearest_url
        routing_url = 'http://127.0.0.1:5000/route/v1/driving/'
        nearest_url = 'http://127.0.0.1:5000/nearest/v1/driving/'
        return routing_url,nearest_url
    
    def get_response(url):
        data = None
        try_count = 0
        while True:
            try_count+=1
            try:
                data=requests.get(url)
                return data.json()
            except requests.exceptions.RequestException as e:
                if try_count > 5:
                    raise e
                time.sleep(1)

    def get_simple_route_reponse(source,dest):
        url="{0}{1},{2};{3},{4}".format(routing_url,source.lon,source.lat,dest.lon,dest.lat)
        return NetworkHandler.get_response(url)
    
    def get_detailed_route_reponse(source,dest):
        url="{0}{1},{2};{3},{4}?steps=true&geometries=geojson".format(routing_url,source.lon,source.lat,dest.lon,dest.lat)
        return NetworkHandler.get_response(url)

    def travel_time(source,destination):
        response = NetworkHandler.get_simple_route_reponse(source,destination)
        return response['routes'][0]['duration']

    def travel_distance(source,destination):
        response = NetworkHandler.get_simple_route_reponse(source,destination)
        return response['routes'][0]['distance']    

    def get_current_location_time(source,destination,starting_time,current_time):
        response = NetworkHandler.get_detailed_route_reponse(source,destination)
        for step in response['routes'][0]['legs'][0]['steps']:
            duration = step['duration']
            starting_time += timedelta(seconds=duration)
            if starting_time >= current_time:
                location = step['geometry']['coordinates'][-1]
                return starting_time,Node(location[1],location[0])

    def get_nearest_node(lat,lon):
        url="{0}{1},{2}".format(nearest_url,lon, lat)
        data = NetworkHandler.get_response(url)
        nearest_node = data['waypoints'][0]['location']
        return nearest_node[1],nearest_node[0]
