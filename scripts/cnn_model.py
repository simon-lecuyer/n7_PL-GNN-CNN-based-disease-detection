import torch
import torch.nn as nn


class DiseaseCNN(nn.Module):
    """
    3D CNN pour predire la carte a t+L+1.
    Input: (B, L, 64, 64)
    Output: (B, 1, 64, 64)
    
    Améliorations légères:
    - 2 couches Conv3D 
    - 1 couche Conv2D supplémentaire
    - Batch Normalization
    - Dropout léger
    """

    def __init__(self, in_frames=5, out_channels=1, hidden_channels=16):
        """
        Args:
            in_frames: nombre de frames temporelles en entree (L)
            out_channels: nombre de canaux de sortie
            hidden_channels: canaux internes (16 = bon compromis)
        """
        super(DiseaseCNN, self).__init__()

        # Couche 1: 1 → hidden_channels
        self.conv3d_1 = nn.Conv3d(1, hidden_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3d_1 = nn.BatchNorm3d(hidden_channels)
        
        # Couche 2: hidden_channels → hidden_channels
        self.conv3d_2 = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3d_2 = nn.BatchNorm3d(hidden_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout3d = nn.Dropout3d(p=0.2)  # Dropout léger

        # Compresser la dimension temporelle
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, 64, 64))

        # Projection 2D vers la carte finale
        self.conv2d = nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1)
        self.bn2d = nn.BatchNorm2d(hidden_channels // 2)
        self.dropout2d = nn.Dropout2d(p=0.2)
        
        self.head = nn.Conv2d(hidden_channels // 2, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: (B, L, 64, 64)
        Returns:
            out: (B, 1, 64, 64)
        """
        x = x.unsqueeze(1)  # (B, 1, L, 64, 64)
        
        # Bloc 1
        x = self.conv3d_1(x)
        x = self.bn3d_1(x)
        x = self.relu(x)
        
        # Bloc 2
        x = self.conv3d_2(x)
        x = self.bn3d_2(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        
        # Compression temporelle
        x = self.temporal_pool(x)  # (B, C, 1, 64, 64)
        x = x.squeeze(2)  # (B, C, 64, 64)
        
        # Projection 2D
        x = self.conv2d(x)
        x = self.bn2d(x)
        x = self.relu(x)
        x = self.dropout2d(x)
        
        out = self.sigmoid(self.head(x))
        return out