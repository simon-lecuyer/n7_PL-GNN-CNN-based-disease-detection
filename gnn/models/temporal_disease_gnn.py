import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv


class TemporalDiseaseGNN(nn.Module):
    """
    Temporal GNN for disease spread prediction.

    Input: sequence of graphs (length = T)
    Output: infection probabilities at next timestep
    """

    def __init__(
        self,
        in_channels=3,
        hidden_dim=64,
        num_layers=2,
        out_channels=1,
        dropout=0.2
    ):
        super().__init__()

        self.dropout = dropout

        # --- Spatial GNN encoder ---
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # --- Temporal model (GRU over node embeddings) ---
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        # --- Prediction head ---
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_channels)
        )

    def spatial_encode(self, x, edge_index):
        """Encode one graph snapshot."""
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, graph_sequence):
        """
        graph_sequence: list of graph dicts (length T)

        Returns:
            prediction (N,1)
        """

        embeddings = []

        for graph in graph_sequence:

            x = graph["node_features"].float()
            edge_index = graph["edges"].long()

            # Fix edge format if needed
            if edge_index.shape[0] != 2:
                edge_index = edge_index.t()

            # Spatial encoding
            h = self.spatial_encode(x, edge_index)  # (N, hidden_dim)
            embeddings.append(h)

        # Stack over time: (T, N, hidden_dim)
        embeddings = torch.stack(embeddings, dim=0)

        # Convert to (N, T, hidden_dim) for GRU
        embeddings = embeddings.permute(1, 0, 2)

        # Temporal aggregation
        out, _ = self.gru(embeddings)

        # Last hidden state = representation of node history
        h_last = out[:, -1, :]

        # Predict infection at next timestep
        pred = self.mlp(h_last)
        pred = torch.sigmoid(pred)

        return pred
