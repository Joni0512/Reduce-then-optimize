from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.handlers.request_handler import RequestHandler
from rtv_solver.handlers.network_handler import NetworkHandler
from rtv_solver.schema.payload_keys import PayloadKeys
from rtv_solver.structure.config import Config

from rtv_solver.pipeline.request_graph_full import RequestGraphFullBuilder
from rtv_solver.pipeline.request_graph_feature_builder import RequestGraphFeatureBuilder
from rtv_solver.pipeline.request_graph_gnn import RequestGraphEdgeGNN
from rtv_solver.pipeline.request_graph_label_builder import RequestGraphLabelBuilder
from rtv_solver.handlers.vehicle_handler import VehicleHandler
from rtv_solver.pipeline.request_graph_vehicle_feature_builder import RequestGraphVehicleFeatureBuilder


# ------------------------------------------------------------
# 1. Payload laden
# ------------------------------------------------------------
payload_path = Path("solutions/li_lim/manifests/lc103.json")
label_mode = "expert"   # "heuristic" oder "expert"
#payload_path = Path("solutions/li_lim/txt_files/lc101.txt")
# payload_path = Path("inputs/localDB_payload_oct.pkl")
# payload_path = Path("inputs/wilson_nc_initial.pkl")

config = Config()
payload = PayloadParser.load_input_data(payload_path)

NetworkHandler.init_from_payload(
    payload=payload,
    server_url=config.SERVER_URL,
)

request_handler = RequestHandler(
    payload[PayloadKeys.REQUESTS],
    config,
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


# ------------------------------------------------------------
# 2. Request-Graph bauen
# ------------------------------------------------------------
# Nodes = Requests
# Edges = Request-Request-Paare

G = RequestGraphFullBuilder.build(requests)
G = RequestGraphFeatureBuilder.add_features(G)

#G = RequestGraphVehicleFeatureBuilder.add_vehicle_and_local_context_features(
#    G,
#    vehicle_handler.vehicles,
#    max_travel_time_seconds=600.0,
#)

node_matrix, edge_index, edge_matrix, node_ids = RequestGraphFeatureBuilder.to_numpy(G)

node_df = pd.DataFrame(
    node_matrix,
    columns=RequestGraphFeatureBuilder.NODE_FEATURES,
)

edge_df = pd.DataFrame(
    edge_matrix,
    columns=RequestGraphFeatureBuilder.EDGE_FEATURES,
)

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())


# ------------------------------------------------------------
# 3. Heuristik-Labels bauen
# ------------------------------------------------------------
# Das sind noch keine echten Labels aus der ILP-Lösung.
# Das ist nur ein Technik-Test:
# Kann das GNN lernen, unsere einfache Regel vorherzusagen?
# ------------------------------------------------------------
# 3. Labels bauen
# ------------------------------------------------------------
# ------------------------------------------------------------
# 3. Labels bauen
# ------------------------------------------------------------

if label_mode == "heuristic":
    labels_np = RequestGraphLabelBuilder.build_heuristic_labels(
        edge_df=edge_df,
        max_pickup_time_difference=3600.0,
        min_direction_similarity=0.0,
        require_pickup_overlap=True,
    )

elif label_mode == "expert":
    labels_np = RequestGraphLabelBuilder.build_expert_labels_from_solution_payload(
        edge_index=edge_index,
        node_ids=node_ids,
        solution_payload=payload,
    )

else:
    raise ValueError(f"Unknown label_mode: {label_mode}")

print("Label mode:", label_mode)
print("Positive labels:", labels_np.sum())
print("Negative labels:", len(labels_np) - labels_np.sum())
print("Positive ratio:", labels_np.mean())

# ------------------------------------------------------------
# 4. Features normalisieren und Tensoren bauen
# ------------------------------------------------------------
# Wichtig:
# Die Rohfeatures haben sehr unterschiedliche Skalen:
# earliest_pickup_time kann z.B. 40000 sein,
# direction_similarity liegt aber zwischen -1 und 1.
# Deshalb normalisieren wir auf ungefähr mean=0, std=1.

node_matrix_norm = (
    (node_df - node_df.mean()) / (node_df.std() + 1e-8)
).to_numpy()

edge_matrix_norm = (
    (edge_df - edge_df.mean()) / (edge_df.std() + 1e-8)
).to_numpy()

x = torch.tensor(node_matrix_norm, dtype=torch.float32)
edge_attr = torch.tensor(edge_matrix_norm, dtype=torch.float32)
labels = torch.tensor(labels_np, dtype=torch.float32)


# ------------------------------------------------------------
# 5. Edge Index von Booking IDs auf Tensor-Indizes mappen
# ------------------------------------------------------------
# edge_index enthält aktuell echte Request IDs / Booking IDs.
# Beispiel: 1129707
#
# PyTorch braucht aber Tensor-Positionen:
# 0, 1, 2, ..., num_nodes-1

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
    dtype=torch.long,
).t().contiguous()


# ------------------------------------------------------------
# 6. Modell, Loss, Optimizer
# ------------------------------------------------------------

model = RequestGraphEdgeGNN(
    node_feature_dim=x.shape[1],
    edge_feature_dim=edge_attr.shape[1],
    hidden_dim=64,
    message_passing_steps=2,
)

# BCEWithLogitsLoss erwartet rohe logits.
# Deshalb sollte request_graph_gnn.py am Ende return logits machen,
# nicht return sigmoid(logits).
criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-2,
)


# ------------------------------------------------------------
# 7. Training mit balanced sampling
# ------------------------------------------------------------
# Problem:
# Es gibt viel mehr negative als positive Edges.
#
# Bei dir ca.:
# positive: 913
# negative: 17423
#
# Wenn wir alle Edges gleichzeitig trainieren,
# lernt das Modell leicht: "alles negativ".
#
# Lösung:
# Pro Epoch nehmen wir:
# - alle positiven Edges
# - gleich viele zufällige negative Edges

epochs = 100

pos_idx = torch.where(labels == 1)[0]
neg_idx = torch.where(labels == 0)[0]

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    logits = model(
        x=x,
        edge_index=edge_index_t,
        edge_attr=edge_attr,
    )

    # Gleich viele negative Beispiele sampeln wie positive.
    sampled_neg_idx = neg_idx[
        torch.randperm(len(neg_idx))[:len(pos_idx)]
    ]

    train_idx = torch.cat([pos_idx, sampled_neg_idx])
    train_idx = train_idx[torch.randperm(len(train_idx))]

    # Loss nur auf dem balanced Trainingssubset.
    loss = criterion(
        logits[train_idx],
        labels[train_idx],
    )

    loss.backward()
    optimizer.step()

    with torch.no_grad():
        scores = torch.sigmoid(logits)
        predictions = scores > 0.5

        # Accuracy auf allen Edges.
        # Achtung: Bei unbalanced Labels ist Accuracy nicht super aussagekräftig.
        correct = (predictions.float() == labels).float().mean().item()
        kept_edges = predictions.sum().item()

        positive_scores = scores[labels == 1]
        negative_scores = scores[labels == 0]

        mean_pos_score = positive_scores.mean().item()
        mean_neg_score = negative_scores.mean().item()

    if epoch % 5 == 0 or epoch == epochs - 1:
        print(
            f"Epoch {epoch:03d} | "
            f"Loss: {loss.item():.4f} | "
            f"Accuracy: {correct:.4f} | "
            f"Mean pos: {mean_pos_score:.4f} | "
            f"Mean neg: {mean_neg_score:.4f} | "
            f"Edges kept @0.5: {kept_edges}/{labels.numel()}"
        )


# ------------------------------------------------------------
# 8. Score-Verteilung nach Training anschauen
# ------------------------------------------------------------

model.eval()

with torch.no_grad():
    logits = model(
        x=x,
        edge_index=edge_index_t,
        edge_attr=edge_attr,
    )
    scores = torch.sigmoid(logits)

edge_df["label"] = labels.numpy()
edge_df["gnn_score"] = scores.numpy()

print("\nFINAL SCORE DISTRIBUTION")
print(edge_df["gnn_score"].describe())

print("\nMEAN SCORE BY LABEL")
print(edge_df.groupby("label")["gnn_score"].mean())

print("\nTHRESHOLD COUNTS")
for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    kept = (edge_df["gnn_score"] > threshold).sum()
    print(f"Edges kept @ {threshold:.1f}: {kept}")

print("\nPRUNING QUALITY")
for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    keep_mask = edge_df["gnn_score"] > threshold

    kept_total = keep_mask.sum()
    kept_positive = ((edge_df["label"] == 1) & keep_mask).sum()
    total_positive = (edge_df["label"] == 1).sum()

    recall = kept_positive / total_positive if total_positive > 0 else 0.0
    reduction = 1.0 - kept_total / len(edge_df)

    print(
        f"Threshold {threshold:.1f} | "
        f"kept {kept_total}/{len(edge_df)} | "
        f"positive kept {kept_positive}/{total_positive} | "
        f"recall {recall:.3f} | "
        f"edge reduction {reduction:.3f}"
    )