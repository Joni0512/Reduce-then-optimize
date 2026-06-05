from pathlib import Path
import pandas as pd
import torch

from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.handlers.request_handler import RequestHandler
from rtv_solver.handlers.network_handler import NetworkHandler
from rtv_solver.schema.payload_keys import PayloadKeys
from rtv_solver.structure.config import Config

from rtv_solver.pipeline.request_graph_full import RequestGraphFullBuilder
from rtv_solver.pipeline.request_graph_feature_builder import RequestGraphFeatureBuilder
from rtv_solver.pipeline.request_graph_gnn import RequestGraphEdgeGNN
from rtv_solver.handlers.vehicle_handler import VehicleHandler
from rtv_solver.pipeline.request_graph_vehicle_feature_builder import RequestGraphVehicleFeatureBuilder


payload_path = Path("inputs/localDB_payload_oct.pkl")

config = Config()
payload = PayloadParser.load_input_data(payload_path)

NetworkHandler.init_from_payload(
    payload=payload,
    server_url=config.SERVER_URL
)

request_handler = RequestHandler(
    payload[PayloadKeys.REQUESTS],
    config
)

requests = request_handler.get_all_requests()

payload_object = PayloadParser.get_payload_object(
    payload,
    dwell_pickup_default=config.DWELL_PICKUP,
    dwell_alight_default=config.DWELL_ALIGHT,
    online=False,
)

vehicle_handler = VehicleHandler(
    payload_object.depot,
    payload_object.driver_runs,
    config,
)

G = RequestGraphFullBuilder.build(requests)
G = RequestGraphFeatureBuilder.add_features(G)

G = RequestGraphVehicleFeatureBuilder.add_vehicle_and_local_context_features(
    G,
    vehicle_handler.vehicles,
    max_travel_time_seconds=600.0,
)

node_matrix, edge_index, edge_matrix, node_ids = RequestGraphFeatureBuilder.to_numpy(G)

print("Requests / nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
print("Node matrix shape:", node_matrix.shape)
print("Edge index shape:", edge_index.shape)
print("Edge matrix shape:", edge_matrix.shape)

node_df = pd.DataFrame(node_matrix, columns=RequestGraphFeatureBuilder.NODE_FEATURES)
edge_df = pd.DataFrame(edge_matrix, columns=RequestGraphFeatureBuilder.EDGE_FEATURES)

print("\nNODE FEATURES HEAD")
print(node_df.head())

print("\nEDGE FEATURES HEAD")
print(edge_df.head())

print("\nNODE FEATURE DISTRIBUTION")
print(node_df.describe().T)

print("\nEDGE FEATURE DISTRIBUTION")
print(edge_df.describe().T)

print("\nPRUNING HEURISTIC COUNTS")
print("Edges total:", len(edge_df))
print("Overlap > 0:", (edge_df["pickup_window_overlap_seconds"] > 0).sum())
print("Time diff <= 1800:", (edge_df["pickup_time_difference"] <= 1800).sum())
print("Time diff <= 3600:", (edge_df["pickup_time_difference"] <= 3600).sum())
print(
    "Overlap > 0 and direction > 0:",
    (
        (edge_df["pickup_window_overlap_seconds"] > 0)
        & (edge_df["direction_similarity"] > 0)
    ).sum()
)


# ------------------------------------------------------------
# GNN TEST
# ------------------------------------------------------------
# Achtung:
# Das Modell ist hier noch NICHT trainiert.
# Die Scores sind deshalb random initialisiert.
# Dieser Test prüft erstmal nur:
#   RequestGraph -> tensors -> GNN -> edge_scores
# ------------------------------------------------------------

x = torch.tensor(node_matrix, dtype=torch.float32)

# Dein edge_index aus to_numpy hat Shape [num_edges, 2].
# Das GNN erwartet Shape [2, num_edges].
#edge_index_t = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
id_to_idx = {
    node_id: idx
    for idx, node_id in enumerate(node_ids)
}

edge_index_reindexed = [
    [id_to_idx[u], id_to_idx[v]]
    for u, v in edge_index
]

edge_index_t = torch.tensor(
    edge_index_reindexed,
    dtype=torch.long
).t().contiguous()

edge_attr = torch.tensor(edge_matrix, dtype=torch.float32)

model = RequestGraphEdgeGNN(
    node_feature_dim=x.shape[1],
    edge_feature_dim=edge_attr.shape[1],
    hidden_dim=64,
    message_passing_steps=2,
)

model.eval()

with torch.no_grad():
    edge_scores = model(x, edge_index_t, edge_attr)

print("\nGNN EDGE SCORES")
print("Edge scores shape:", edge_scores.shape)
print("Min score:", edge_scores.min().item())
print("Mean score:", edge_scores.mean().item())
print("Max score:", edge_scores.max().item())

keep_mask = edge_scores > 0.5
print("Edges kept @ 0.5:", keep_mask.sum().item())
print("Edges total:", edge_scores.numel())


# Scores in DataFrame packen, damit du sie mit den anderen Edge-Features vergleichen kannst.
edge_df["gnn_score"] = edge_scores.detach().cpu().numpy()

print("\nGNN SCORE DISTRIBUTION")
print(edge_df["gnn_score"].describe())

print("\nEDGE FEATURES + GNN SCORE HEAD")
print(edge_df.head())

print("\nGNN SCORE BY SIMPLE HEURISTICS")
print(
    "Mean score where overlap > 0:",
    edge_df.loc[
        edge_df["pickup_window_overlap_seconds"] > 0,
        "gnn_score"
    ].mean()
)

print(
    "Mean score where overlap == 0:",
    edge_df.loc[
        edge_df["pickup_window_overlap_seconds"] == 0,
        "gnn_score"
    ].mean()
)

print(
    "Mean score where direction > 0:",
    edge_df.loc[
        edge_df["direction_similarity"] > 0,
        "gnn_score"
    ].mean()
)

print(
    "Mean score where direction <= 0:",
    edge_df.loc[
        edge_df["direction_similarity"] <= 0,
        "gnn_score"
    ].mean()
)

print(
    "Mean score where overlap > 0 and direction > 0:",
    edge_df.loc[
        (edge_df["pickup_window_overlap_seconds"] > 0)
        & (edge_df["direction_similarity"] > 0),
        "gnn_score"
    ].mean()
)


print("\nGNN SCORE THRESHOLD COUNTS")
for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    kept = (edge_df["gnn_score"] > threshold).sum()
    print(f"Edges kept @ {threshold:.1f}: {kept}")