from pathlib import Path

from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.handlers.request_handler import RequestHandler
from rtv_solver.handlers.network_handler import NetworkHandler
from rtv_solver.schema.payload_keys import PayloadKeys
from rtv_solver.structure.config import Config

from rtv_solver.pipeline.request_graph_full import RequestGraphFullBuilder

import networkx as nx


payload_path = Path("inputs/full_requests_payload.pkl")

config = Config()

payload = PayloadParser.load_input_data(payload_path)

print("Payload keys:", payload.keys())
print("Raw requests:", len(payload["requests"]))
print("Driver runs:", len(payload["driver_runs"]))

NetworkHandler.init_from_source(
    server_url="http://127.0.0.1:5001/",
    euclidean=True,
)

request_handler = RequestHandler(
    payload[PayloadKeys.REQUESTS],
    config
)

requests = request_handler.get_all_requests()

print("Parsed requests:", len(requests))

G = RequestGraphFullBuilder.build(requests)

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
print("Density:", nx.density(G))

expected_edges = len(requests) * (len(requests) - 1) // 2
print("Expected edges:", expected_edges)

if G.number_of_edges() == expected_edges:
    print("OK: Full request graph was built correctly.")
else:
    print("WARNING: Edge count does not match complete graph.")