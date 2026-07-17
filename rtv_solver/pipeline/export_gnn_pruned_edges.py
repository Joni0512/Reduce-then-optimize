"""
Scores a request-graph GNN checkpoint over a manifest's requests and exports
the kept edges as a CSV in the format RollingHorizon's ilp_full.cpp
apply_gnn_pruning() reads via GNN_PRUNING_FILE:

    header row (ignored), then one row per kept pair: "request_i,request_j"

Usage:
    python3 -m rtv_solver.pipeline.export_gnn_pruned_edges \
        --manifest solutions/nyc/manifests/nyc_morning500_mc3_fixed.json \
        --checkpoint outputs/models_nyc/rgnn_nyc_morning500_mc3_fixed_pw5_v2/rgnn_nyc_morning500_mc3_fixed_pw5_v2_best_val_f3.pt \
        --threshold 0.3 \
        --out /Users/joni/Desktop/Masterarbeit/external_repos/RollingHorizon/results/nyc_morning500_mc3_fixed/pruned_keep_edges_0.30.csv

Then run RollingHorizon with:
    USE_GNN_PRUNING true GNN_PRUNING_FILE results/nyc_morning500_mc3_fixed/pruned_keep_edges_0.30.csv
"""

import argparse
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
from rtv_solver.pipeline.request_graph_gnn_v2 import RequestGraphEdgeGNNv2
from rtv_solver.pipeline.request_graph_pruner import _detect_gnn_version

HIDDEN_DIM = 64


def load_model(checkpoint_path: Path, node_feature_dim: int, edge_feature_dim: int):
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    version = _detect_gnn_version(state_dict)

    if version == "v2":
        layer_indices = {
            int(key.split(".")[1])
            for key in state_dict
            if key.startswith("message_mlps.")
        }
        num_layers = max(layer_indices) + 1
        model = RequestGraphEdgeGNNv2(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=HIDDEN_DIM,
            num_layers=num_layers,
        )
    else:
        model = RequestGraphEdgeGNN(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=HIDDEN_DIM,
            message_passing_steps=2,
        )

    model.load_state_dict(state_dict)
    model.eval()
    return model, version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="manifest JSON (our own format) to score")
    parser.add_argument("--checkpoint", required=True, help="trained GNN pruner .pt checkpoint")
    parser.add_argument("--threshold", type=float, required=True, help="keep edges with sigmoid score >= threshold")
    parser.add_argument("--out", required=True, help="output CSV path (RollingHorizon GNN_PRUNING_FILE target)")
    args = parser.parse_args()

    config = Config()
    payload = PayloadParser.load_input_data(Path(args.manifest))
    NetworkHandler.init_from_payload(payload=payload, server_url=config.SERVER_URL)
    requests = RequestHandler(payload[PayloadKeys.REQUESTS], config).get_all_requests()

    G = RequestGraphFullBuilder.build(requests)
    G = RequestGraphFeatureBuilder.add_features(G)
    node_matrix, edge_index, edge_matrix, node_ids = RequestGraphFeatureBuilder.to_numpy(G)

    node_df = pd.DataFrame(node_matrix, columns=RequestGraphFeatureBuilder.NODE_FEATURES)
    edge_df = pd.DataFrame(edge_matrix, columns=RequestGraphFeatureBuilder.EDGE_FEATURES)
    node_matrix_norm = ((node_df - node_df.mean()) / (node_df.std() + 1e-8)).to_numpy()
    edge_matrix_norm = ((edge_df - edge_df.mean()) / (edge_df.std() + 1e-8)).to_numpy()

    id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    edge_index_reindexed = [[id_to_idx[u], id_to_idx[v]] for u, v in edge_index]

    x = torch.tensor(node_matrix_norm, dtype=torch.float32)
    edge_attr = torch.tensor(edge_matrix_norm, dtype=torch.float32)
    edge_index_t = torch.tensor(edge_index_reindexed, dtype=torch.long).t().contiguous()

    model, version = load_model(Path(args.checkpoint), x.shape[1], edge_attr.shape[1])
    print(f"Loaded {version} checkpoint: {args.checkpoint}")

    with torch.no_grad():
        scores = torch.sigmoid(model(x=x, edge_index=edge_index_t, edge_attr=edge_attr)).numpy()

    # RequestGraphFullBuilder builds an undirected nx.Graph via combinations(), so
    # edge_index already has exactly one row per unordered pair - sort() here just
    # normalizes (u, v) ordering to match apply_gnn_pruning's pair_key() convention,
    # not deduping anything.
    kept_pairs = set()
    for (u, v), score in zip(edge_index, scores):
        if score >= args.threshold:
            kept_pairs.add(tuple(sorted((int(u), int(v)))))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("request_i,request_j\n")
        for a, b in sorted(kept_pairs):
            f.write(f"{a},{b}\n")

    print(f"Total edges: {len(edge_index)} | kept (threshold={args.threshold}): {len(kept_pairs)}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
