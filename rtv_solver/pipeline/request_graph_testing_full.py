from pathlib import Path

from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.handlers.request_handler import RequestHandler
from rtv_solver.handlers.network_handler import NetworkHandler
from rtv_solver.schema.payload_keys import PayloadKeys
from rtv_solver.structure.config import Config

from rtv_solver.pipeline.request_graph_full import RequestGraphFullBuilder
from rtv_solver.pipeline.request_graph_feature_builder import RequestGraphFeatureBuilder

import matplotlib.pyplot as plt
import networkx as nx


payload_path = Path("inputs/localDB_payload_oct.pkl")

config = Config()

payload = PayloadParser.load_input_data(payload_path)

print(payload.keys())
print(len(payload["requests"]))
print(len(payload["driver_runs"]))

NetworkHandler.init_from_payload(
    payload,
    server_url=config.SERVER_URL
)

config = Config()
payload = PayloadParser.load_input_data(payload_path)

NetworkHandler.init_from_payload(
    payload,
    server_url=config.SERVER_URL
)

request_handler = RequestHandler(
    payload[PayloadKeys.REQUESTS],
    config
)

requests = request_handler.get_all_requests()

# FULL REQUEST GRAPH
G = RequestGraphFullBuilder.build(requests)

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
print("Density:", nx.density(G))

print("Connected components:")
for comp in nx.connected_components(G):
    print(sorted(comp))

# VISUALIZATION
plt.figure(figsize=(10, 10))

pos = nx.spring_layout(
    G,
    seed=42
)

nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=500,
    node_color="skyblue",
    font_size=8,
    font_color="black"
)

plt.title(
    f"Full Request Graph ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)"
)

plt.show()