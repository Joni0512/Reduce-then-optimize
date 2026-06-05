import networkx as nx
from itertools import combinations

class RequestGraphBuilder:
    @staticmethod
    def build_same_route_graph(requests):
        G = nx.Graph()

        for req in requests:
            G.add_node(req.id, request=req)

        for r1, r2 in combinations(requests, 2):
            same_pickup = (
                r1.origin.lat == r2.origin.lat
                and r1.origin.lon == r2.origin.lon
            )
            same_dropoff = (
                r1.destination.lat == r2.destination.lat
                and r1.destination.lon == r2.destination.lon
            )

            if same_pickup and same_dropoff:
                G.add_edge(r1.id, r2.id)

        return G
