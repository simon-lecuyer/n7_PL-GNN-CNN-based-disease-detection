import torch
import torch.nn as nn


class DiseaseCNN(nn.Module):
    """
    3D CNN minimal pour predire la carte a t+L+1.
    Input: (B, L, 64, 64)
    Output: (B, 1, 64, 64)
    """

    def __init__(self, in_frames=5, out_channels=1, hidden_channels=8):
        """
        Args:
            in_frames: nombre de frames temporelles en entree (L)
            out_channels: nombre de canaux de sortie
            hidden_channels: canaux internes (petit = modele simple)
        """
        super(DiseaseCNN, self).__init__()

        self.conv3d = nn.Conv3d(1, hidden_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.relu = nn.ReLU(inplace=True)

        # Compresser la dimension temporelle a 1 en gardant la resolution spatiale.
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, 64, 64))

        # Projection 2D vers la carte finale.
        self.head = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: (B, L, 64, 64)
        Returns:
            out: (B, 1, 64, 64)
        """
        x = x.unsqueeze(1)  # (B, 1, L, 64, 64)
        x = self.relu(self.conv3d(x))
        x = self.temporal_pool(x)  # (B, C, 1, 64, 64)
        x = x.squeeze(2)  # (B, C, 64, 64)
        out = self.sigmoid(self.head(x))
        return out