from pathlib import Path
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.handlers.request_handler import RequestHandler
from rtv_solver.handlers.network_handler import NetworkHandler
from rtv_solver.schema.payload_keys import PayloadKeys
from rtv_solver.structure.config import Config
from rtv_solver.pipeline.request_graph import RequestGraphBuilder
import matplotlib.pyplot as plt
import networkx as nx

payload_path = Path("inputs/test_nc/ttm/test_10r_1v_repeat6_simple.pkl")

config = Config()
payload = PayloadParser.load_input_data(payload_path)

NetworkHandler.init_from_payload(payload, server_url=config.SERVER_URL)

request_handler = RequestHandler(payload[PayloadKeys.REQUESTS], config)
requests = request_handler.get_all_requests()

G = RequestGraphBuilder.build_same_route_graph(requests)

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
print("Density:", __import__("networkx").density(G))

print("Connected components:")
for comp in __import__("networkx").connected_components(G):
    print(sorted(comp))

# Visualisierung des Graphen
plt.figure(figsize=(8, 8))
pos = nx.spring_layout(G, seed=42)  # Für eine stabile Darstellung
nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", font_size=10, font_color="black")
plt.title("Request-Request-Graph")
plt.show()