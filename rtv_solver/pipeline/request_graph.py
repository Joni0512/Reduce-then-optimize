import networkx as nx
from itertools import combinations

class RequestGraphBuilder:
    """
    Builds a simple request-request graph for diagnostic experiments.

    Each node represents one transportation request.
    An undirected edge connects two requests if they share the same
    pickup and drop-off locations. This builder is mainly useful for
    sanity checks on repeated or identical requests.

    For the actual pruning pipeline, the full request graph builder is used,
    where edges represent all candidate request pairs and are enriched with
    temporal, spatial, and compatibility features.
    """
    @staticmethod
    def build_same_route_graph(requests):
        """
        Construct a graph connecting requests with identical origin and
        destination coordinates.

        Parameters
        ----------
        requests:
            Iterable of request objects. Each request must expose
            `id`, `origin`, and `destination` attributes.

        Returns
        -------
        nx.Graph
            Undirected graph with request IDs as node IDs and the original
            request object stored as a node attribute.
        """
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
