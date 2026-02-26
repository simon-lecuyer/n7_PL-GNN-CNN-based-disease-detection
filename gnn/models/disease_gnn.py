import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv


class DiseaseGNN(nn.Module):
    """
    Graph Neural Network for disease propagation prediction.

    Node features:
        [infection, x_coordinate, y_coordinate]

    Output:
        infection probability for each node in [0,1]
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 3,
        out_channels: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # --- Graph convolution layers ---
        self.convs = nn.ModuleList()

        # First layer: input → hidden
        self.convs.append(GCNConv(in_channels, hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Last layer: hidden → hidden (before final MLP)
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # --- Final prediction head ---
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_channels)
        )

    def forward(self, x, edge_index):
        """
        Forward pass.

        Args:
            x: Node feature matrix (N, in_channels)
            edge_index: Graph connectivity (2, E)

        Returns:
            out: Infection prediction per node (N, 1)
        """

        # Message passing through GCN layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Final node-wise prediction
        out = self.mlp(x)

        # Output in [0,1]
        out = torch.sigmoid(out)

        return out
