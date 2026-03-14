from rtv_solver.structure.node import Node
import requests
import time
import numpy as np
from multiprocessing.sharedctypes import RawArray, RawValue
import ctypes
import math

from rtv_solver.schema.payload_keys import PayloadKeys
from rtv_solver.util.logger import BASIC_LOGGER
import logging

console_logger = logging.getLogger(BASIC_LOGGER)

# ---- Globals predeclared to avoid NameError in worker processes ----
SERVER_BASED = None
EUCLIDEAN = None
routing_url = None
nearest_url = None
table_url = None
session = None
travel_time_matrix = None
no_of_nodes = None

class NetworkHandler:
    NODE_INDEX = 0
    node_data = []

    @staticmethod
    def init_from_payload(payload: dict, server_url=None, euclidean=False) -> bool:
        """
        Initialize the network from a payload and indicate if server-side matrix
        precomputation is required.

        Returns:
            bool: True when no precomputed matrix is provided in payload.
        """
        tt_matrix = payload.get(PayloadKeys.TIME_MATRIX)
        NetworkHandler.init_from_source(
            server_url=server_url,
            tt_matrix=tt_matrix,
            euclidean=euclidean,
        )
        return tt_matrix is None

    @staticmethod
    def init_from_source(server_url=None, tt_matrix=None, euclidean=False):
        if tt_matrix is not None:
            return NetworkHandler.init(False, tt_matrix=tt_matrix, euclidean=euclidean)
        return NetworkHandler.init(True, server_url=server_url, euclidean=euclidean)

    @staticmethod
    def needs_runtime_matrix_build() -> bool:
        """
        True when travel times must be built from discovered nodes at runtime.
        """
        global SERVER_BASED, EUCLIDEAN
        server_mode = SERVER_BASED is not None and SERVER_BASED.value
        euclidean_mode = EUCLIDEAN is not None and EUCLIDEAN.value
        return server_mode or euclidean_mode

    @staticmethod
    def check_server_availability(server_url):
        try:
            response = requests.get(server_url)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException:
            return False

    @staticmethod
    def init(server_based, server_url=None, tt_matrix=None, euclidean=False):
        """
        For liLim, Euclidean must be True
        For Sartori, you just need the time matrix as it is part of the dataset.
        """
        global SERVER_BASED, routing_url, nearest_url, table_url, session
        global travel_time_matrix, no_of_nodes, EUCLIDEAN

        NetworkHandler.NODE_INDEX = 0
        NetworkHandler.node_data = []
        SERVER_BASED = RawValue(ctypes.c_bool, server_based)
        EUCLIDEAN = RawValue(ctypes.c_bool, euclidean)
        if SERVER_BASED.value:
            routing_url = server_url + 'route/v1/driving/'
            nearest_url = server_url + 'nearest/v1/driving/'
            table_url = server_url + 'table/v1/driving/'
            session = requests.Session()
            return routing_url, nearest_url, session, table_url, SERVER_BASED
        elif EUCLIDEAN.value: # NOTE this probably does not work as expected
            return None, None, None, None, SERVER_BASED, EUCLIDEAN
        else: # if time matrix is provided in the payload
            travel_time_matrix = np.array(tt_matrix)
            no_of_nodes = RawValue(ctypes.c_uint, travel_time_matrix.shape[0])
            travel_time_matrix = RawArray(
                np.ctypeslib.as_ctypes_type(travel_time_matrix.dtype),
                travel_time_matrix.flatten()
            )
            return travel_time_matrix, no_of_nodes, SERVER_BASED, EUCLIDEAN

    @staticmethod
    def get_next_node_id(lat: float, lon: float) -> int:
        NetworkHandler.node_data.append({"lat": lat, "lon": lon})
        NetworkHandler.NODE_INDEX += 1
        return NetworkHandler.NODE_INDEX - 1

    @staticmethod
    def initialize_travel_time_matrix():
        """
        calculates all possible travel times between all nodes available nodes from the `table_url` API and stores them in a shared memory matrix for fast access during the optimization.

        Whenever we call this, we have considered all request nodes and vehicles, so we can calculate all possible travel times between all nodes and do not have to call the backend server for each calculation.
        """
        global SERVER_BASED, EUCLIDEAN, travel_time_matrix, no_of_nodes

        num_nodes = len(NetworkHandler.node_data)
        travel_time_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float64)
        MAX_NUM_COORD = 50

        if EUCLIDEAN.value:
            for i in range(num_nodes):
                for j in range(num_nodes):
                    travel_time_matrix[i, j] = np.linalg.norm(
                        np.array([NetworkHandler.node_data[i]['lon'], NetworkHandler.node_data[i]['lat']]) - 
                        np.array([NetworkHandler.node_data[j]['lon'], NetworkHandler.node_data[j]['lat']]))
            SERVER_BASED = RawValue(ctypes.c_bool, False)
            no_of_nodes = RawValue(ctypes.c_uint, travel_time_matrix.shape[0])
            travel_time_matrix = RawArray(
                np.ctypeslib.as_ctypes_type(travel_time_matrix.dtype),
                travel_time_matrix.flatten()
            )
            return travel_time_matrix, no_of_nodes, SERVER_BASED, EUCLIDEAN

        coordinates = [
            f"{node['lon']},{node['lat']}" for node in NetworkHandler.node_data
        ]

        iterations = math.ceil(num_nodes / MAX_NUM_COORD)
        for i in range(iterations):
            for j in range(iterations):
                origins = coordinates[i * MAX_NUM_COORD:(i + 1) * MAX_NUM_COORD]
                destinations = coordinates[j * MAX_NUM_COORD:(j + 1) * MAX_NUM_COORD]
                origin_indices = [str(k) for k in range(len(origins))]
                destination_indices = [
                    str(len(origins) + k) for k in range(len(destinations))
                ]
                url = f"{table_url}{';'.join(origins + destinations)}" \
                      f"?sources={';'.join(origin_indices)}" \
                      f"&destinations={';'.join(destination_indices)}"
                data = NetworkHandler.get_response(url)
                matrix = np.array(data['durations'])
                travel_time_matrix[
                    i * MAX_NUM_COORD:(i + 1) * MAX_NUM_COORD,
                    j * MAX_NUM_COORD:(j + 1) * MAX_NUM_COORD
                ] = matrix

        no_of_nodes = RawValue(ctypes.c_uint, travel_time_matrix.shape[0])
        SERVER_BASED = RawValue(ctypes.c_bool, False)
        travel_time_matrix = RawArray(
            np.ctypeslib.as_ctypes_type(travel_time_matrix.dtype),
            travel_time_matrix.flatten()
        )
        return travel_time_matrix, no_of_nodes, SERVER_BASED, EUCLIDEAN

    @staticmethod
    def get_response(url):
        global session
        max_retries = 5
        retry_delay_s = 1
        timeout_s = 10

        if session is None:
            raise RuntimeError(
                "NetworkHandler session is not initialized. Call NetworkHandler.init_from_source() or NetworkHandler.init_from_payload() first; depending on whether you provide the time matrix in the payload or not"
            )

        last_exception = None
        for try_count in range(1, max_retries + 1):
            try:
                resp = session.get(url, timeout=timeout_s)
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.RequestException as e:
                last_exception = e
                console_logger.warning(
                    "Request failed (%s/%s) for URL %s: %s",
                    try_count,
                    max_retries,
                    url,
                    e,
                )
            except ValueError as e:
                last_exception = e
                body_preview = ""
                if "resp" in locals():
                    body_preview = resp.text[:300]
                console_logger.warning(
                    "Invalid JSON (%s/%s) for URL %s. Response preview: %r",
                    try_count,
                    max_retries,
                    url,
                    body_preview,
                )

            if try_count < max_retries:
                time.sleep(retry_delay_s)

        raise RuntimeError(
            f"Failed to get valid response from {url} after {max_retries} attempts. \n"
            f"Last error: {last_exception}" ) from last_exception

    @staticmethod
    def get_simple_route_reponse(source: Node, dest: Node) -> dict:
        url = f"{routing_url}{source.lon},{source.lat};{dest.lon},{dest.lat}"
        return NetworkHandler.get_response(url)

    @staticmethod
    def get_detailed_route_reponse(source: Node, dest: Node) -> dict:
        url = f"{routing_url}{source.lon},{source.lat};{dest.lon},{dest.lat}" \
              "?steps=true&geometries=geojson"
        return NetworkHandler.get_response(url)

    @staticmethod
    def get_location(source: Node, destination: Node) -> int:
        return int(source.node_id * no_of_nodes.value + destination.node_id)

    @staticmethod
    def travel_time(source: Node, destination: Node) -> float:
        if SERVER_BASED is None:
            raise RuntimeError("NetworkHandler.init() must be called before travel_time()")
        if SERVER_BASED.value:
            response = NetworkHandler.get_simple_route_reponse(source, destination)
            return response['routes'][0]['duration']
        return travel_time_matrix[NetworkHandler.get_location(source, destination)]

    @staticmethod
    def travel_time_from_node_indices(source: Node, destination: Node) -> float:
        return travel_time_matrix[int(source * no_of_nodes.value + destination)]

    @staticmethod
    def travel_distance(source: Node, destination: Node) -> float:
        if SERVER_BASED is None:
            raise RuntimeError("NetworkHandler.init() must be called before travel_distance()")
        if SERVER_BASED.value:
            response = NetworkHandler.get_simple_route_reponse(source, destination)
            return response['routes'][0]['distance']
        return travel_time_matrix[NetworkHandler.get_location(source, destination)]

    @staticmethod
    def get_current_location_time(source: Node, destination: Node, starting_time: int, current_time: int) -> tuple[int, Node]:
        """
        specifically tracks the actual geometry of the route and the specific times where positions are reached

        :return int: time that the location is reached
        :return Node: location somewhere on the route
        """
        # TODO check how to use this code when the server is not available

        # NOTE NEW VERSION (has not been tested yet, but seems cleaner)
        # response = NetworkHandler.get_detailed_route_reponse(source, destination)
        # if current_time <= starting_time:
        #     return starting_time, source
        # steps = response['routes'][0]['legs'][0]['steps']
        # if not steps:
        #     return starting_time, source
        
        # t = starting_time
        # current_location = source
        # for step in steps:
        #     duration = step["duration"]
        #     t += duration
        #     lon, lat = step["geometry"]["coordinates"][-1]
        #     current_location = Node(lat, lon)

        #     if t >= current_time:
        #         return t, current_location  
        # return t, current_location
        
        # OLD VERSION (it works, so keep if for now)
        response = NetworkHandler.get_detailed_route_reponse(source, destination)
        current_location = None
        for step in response['routes'][0]['legs'][0]['steps']:
            duration = step['duration']
            starting_time += duration
            location = step['geometry']['coordinates'][-1]
            current_location = Node(location[1], location[0])
            if starting_time >= current_time: 
                # no guarantee for reaching a point before current_time, but as close as possible we step out
                return starting_time, current_location
        return starting_time, current_location

    @staticmethod
    def get_nearest_node(lat: float, lon: float) -> tuple[float, float]:
        url = f"{nearest_url}{lon},{lat}"
        data = NetworkHandler.get_response(url)
        nearest_node = data['waypoints'][0]['location']
        return nearest_node[1], nearest_node[0]

    @staticmethod
    def are_nodes_equal(node1: Node, node2: Node) -> bool:
        return node1.lat == node2.lat and node1.lon == node2.lon

    @staticmethod
    def get_travel_time_matrix(nodes) -> tuple[np.array, dict[tuple[float, float]]]:
        if SERVER_BASED and SERVER_BASED.value:
            coordinates = []
            node_indices = {}
            index = 0
            for node in nodes:
                coordinates.append(f"{node.lon},{node.lat}")
                node_indices[(node.lon, node.lat)] = index
                index += 1
            url = f"{table_url}{';'.join(coordinates)}"
            data = NetworkHandler.get_response(url)
            return np.array(data['durations']), node_indices
        return None, None

    @staticmethod
    def travel_time_from_matrix(node1: Node, node2: Node, matrix: np.array, node_indices: dict[tuple[float, float]]) -> float:
        if SERVER_BASED and SERVER_BASED.value:
            index1 = node_indices[(node1.lon, node1.lat)]
            index2 = node_indices[(node2.lon, node2.lat)]
            return matrix[index1, index2]
        return NetworkHandler.travel_time(node1, node2)