import networkx as nx
from itertools import combinations


class RequestGraphFullBuilder:

    @staticmethod
    def build(requests):

        G = nx.Graph()

        for req in requests:
            G.add_node(
                req.id,
                request=req
            )

        for r1, r2 in combinations(requests, 2):
            G.add_edge(
                r1.id,
                r2.id
            )

        return G