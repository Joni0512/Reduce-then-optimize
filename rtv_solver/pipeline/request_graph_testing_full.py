from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.handlers.request_handler import RequestHandler
from rtv_solver.handlers.network_handler import NetworkHandler
from rtv_solver.schema.payload_keys import PayloadKeys
from rtv_solver.structure.config import Config

from rtv_solver.pipeline.request_graph_full import RequestGraphFullBuilder
from rtv_solver.pipeline.request_graph_feature_builder import RequestGraphFeatureBuilder
from rtv_solver.pipeline.request_graph_label_builder import RequestGraphLabelBuilder


payload_path = Path("solutions/li_lim/manifests/lc102.json")

config = Config()
payload = PayloadParser.load_input_data(payload_path)

print("Payload keys:", payload.keys())
print("Requests:", len(payload[PayloadKeys.REQUESTS]))
print("Driver runs:", len(payload[PayloadKeys.DRIVERS]))

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
G = RequestGraphFeatureBuilder.add_features(G)

node_matrix, edge_index, edge_matrix, node_ids = RequestGraphFeatureBuilder.to_numpy(G)

edge_df = pd.DataFrame(
    edge_matrix,
    columns=RequestGraphFeatureBuilder.EDGE_FEATURES,
)

labels = RequestGraphLabelBuilder.build_expert_labels_from_solution_payload(
    edge_index=edge_index,
    node_ids=node_ids,
    solution_payload=payload,
)

edge_df["expert_label"] = labels

print("\nGRAPH")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
print("Density:", nx.density(G))

print("\nEXPERT LABELS")
print("Positive labels:", labels.sum())
print("Negative labels:", len(labels) - labels.sum())
print("Positive ratio:", labels.mean())

print("\nEDGE FEATURES + LABELS HEAD")
print(edge_df.head())

print("\nLABEL DISTRIBUTION")
print(edge_df["expert_label"].value_counts())

print("\nMEAN EDGE FEATURES BY LABEL")
print(edge_df.groupby("expert_label").mean())

# VISUALIZATION
plt.figure(figsize=(10, 10))

pos = nx.spring_layout(G, seed=42)

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