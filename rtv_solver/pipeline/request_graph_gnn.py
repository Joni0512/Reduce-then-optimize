import torch
import torch.nn as nn
import torch.nn.functional as F


class RequestGraphEdgeGNN(nn.Module):
    """
    GNN für Request-Graph-Pruning.

    Input:
        x:
            Node Features, also Request-Features
            Shape: [num_nodes, node_feature_dim]
            Beispiel: [192, 8]

        edge_index:
            Kantenliste
            Shape: [2, num_edges]
            edge_index[0] = source nodes
            edge_index[1] = target nodes

        edge_attr:
            Edge Features, also Request-Request-Features
            Shape: [num_edges, edge_feature_dim]
            Beispiel: [18336, 6]

    Output:
        edge_scores:
            Ein Score pro Edge
            Shape: [num_edges]
            Werte zwischen 0 und 1
            Hoher Score = Kante behalten
            Niedriger Score = Kante prunen
    """

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int = 64,
        message_passing_steps: int = 2,
    ):
        super().__init__()

        self.message_passing_steps = message_passing_steps

        # Wandelt rohe Request-Features in interne Node-Embeddings um.
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Wandelt rohe Request-Request-Features in interne Edge-Embeddings um.
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Message-Netz:
        # Für jede Kante wird eine Nachricht berechnet aus:
        # source node embedding + target node embedding + edge embedding.
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Node-Update-Netz:
        # Aktualisiert jeden Node anhand seiner alten Repräsentation
        # und der aggregierten Nachrichten seiner Nachbarn.
        self.node_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Edge-Scoring-Head:
        # Am Ende bekommt jede Kante einen Score.
        self.edge_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Führt das GNN aus.
        """

        # Sicherstellen, dass edge_index richtig orientiert ist:
        # Wir wollen Shape [2, num_edges].
        if edge_index.dim() != 2:
            raise ValueError("edge_index must be a 2D tensor.")

        if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
            edge_index = edge_index.t().contiguous()

        if edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges].")

        # Node- und Edge-Features encoden.
        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)

        src = edge_index[0]
        dst = edge_index[1]

        num_nodes = x.shape[0]

        for _ in range(self.message_passing_steps):
            # Für jede Kante holen wir:
            # h_src = Embedding des Start-Requests
            # h_dst = Embedding des Ziel-Requests
            h_src = h[src]
            h_dst = h[dst]

            # Nachricht pro Kante berechnen.
            message_input = torch.cat([h_src, h_dst, e], dim=-1)
            messages = self.message_mlp(message_input)

            # Nachrichten auf Ziel-Nodes aufsummieren.
            aggregated = torch.zeros_like(h)
            aggregated.index_add_(0, dst, messages)

            # Da dein Graph ungerichtet ist, schicken wir die Nachricht auch zurück.
            reverse_aggregated = torch.zeros_like(h)
            reverse_aggregated.index_add_(0, src, messages)

            aggregated = aggregated + reverse_aggregated

            # Durchschnitt statt Summe: verhindert, dass Nodes mit vielen Nachbarn
            # automatisch riesige Werte bekommen.
            degree = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
            degree.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
            degree.index_add_(0, src, torch.ones_like(src, dtype=x.dtype))
            degree = degree.clamp(min=1.0).unsqueeze(-1)

            aggregated = aggregated / degree

            # Node-Embeddings aktualisieren.
            update_input = torch.cat([h, aggregated], dim=-1)
            h = self.node_update(update_input)

        # Nach Message Passing: Edge Scores berechnen.
        #h_src = h[src]
        #h_dst = h[dst]

        #edge_score_input = torch.cat([h_src, h_dst, e], dim=-1)
        #logits = self.edge_scorer(edge_score_input).squeeze(-1)

        # Sigmoid macht daraus Werte zwischen 0 und 1.
        #edge_scores = torch.sigmoid(logits)

        #return edge_scores

        # Nach Message Passing: Edge Scores berechnen.
        h_src = h[src]
        h_dst = h[dst]

        edge_score_input = torch.cat([h_src, h_dst, e], dim=-1)

        # Rohe Scores (Logits)
        logits = self.edge_scorer(edge_score_input).squeeze(-1)

        # WICHTIG:
        # Kein sigmoid hier!
        # BCEWithLogitsLoss macht das intern selbst.
        return logits