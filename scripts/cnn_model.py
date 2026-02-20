import torch
import torch.nn as nn


class DiseaseCNN(nn.Module):
    """
    3D CNN temporel avec encoder profond, bottleneck et skip connections.
    Input: (B, L, 64, 64)  - L frames temporelles
    Output: (B, 1) - prédiction scalaire au temps t+L+1
    """
    
    def __init__(self, in_frames=5, out_channels=1):
        super(DiseaseCNN, self).__init__()
        
        # Encoder Block 1: (B, 1, L, 64, 64) → (B, 16, L, 32, 32)
        self.enc1_conv = nn.Conv3d(1, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.enc1_relu = nn.ReLU(inplace=True)
        self.enc1_pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dropout1 = nn.Dropout3d(0.2)
        
        # Encoder Block 2: (B, 16, L, 32, 32) → (B, 32, L, 16, 16)
        self.enc2_conv = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.enc2_relu = nn.ReLU(inplace=True)
        self.enc2_pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dropout2 = nn.Dropout3d(0.25)
        
        # Encoder Block 3: (B, 32, L, 16, 16) → (B, 64, L, 8, 8)
        self.enc3_conv = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.enc3_relu = nn.ReLU(inplace=True)
        self.enc3_pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dropout3 = nn.Dropout3d(0.3)
        
        # Bottleneck: (B, 64, L, 8, 8) → (B, 128, L, 8, 8)
        self.bottleneck_conv = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bottleneck_relu = nn.ReLU(inplace=True)
        self.dropout_bottleneck = nn.Dropout3d(0.35)
        
        # Skip connection adaptor: encoder3 (64 channels) + bottleneck (128) → 128
        # On va concaténer enc3 + bottleneck → 192 channels → réduire à 128
        self.skip_conv = nn.Conv3d(64 + 128, 128, kernel_size=(1, 1, 1))
        self.skip_relu = nn.ReLU(inplace=True)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout_pool = nn.Dropout(0.3)
        
        # MLP Head pour prédiction scalaire
        self.fc1 = nn.Linear(128, 256)
        self.fc1_relu = nn.ReLU(inplace=True)
        self.fc_dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(256, 128)
        self.fc2_relu = nn.ReLU(inplace=True)
        self.fc_dropout2 = nn.Dropout(0.2)
        
        self.fc_out = nn.Linear(128, 1)
        
    def forward(self, x):
        """
        Forward pass avec skip connections du bottleneck vers encoder3
        Args:
            x: (B, L, 64, 64) - L frames temporelles
        Returns:
            out: (B, 1) - prédiction scalaire
        """
        # Ajouter dimension channel: (B, L, 64, 64) → (B, 1, L, 64, 64)
        x = x.unsqueeze(1)
        
        # ENCODER BLOCKS
        # Block 1
        e1 = self.enc1_relu(self.enc1_conv(x))
        e1_pool = self.enc1_pool(e1)
        e1_pool = self.dropout1(e1_pool)
        
        # Block 2
        e2 = self.enc2_relu(self.enc2_conv(e1_pool))
        e2_pool = self.enc2_pool(e2)
        e2_pool = self.dropout2(e2_pool)
        
        # Block 3
        e3 = self.enc3_relu(self.enc3_conv(e2_pool))
        e3_pool = self.enc3_pool(e3)
        e3_pool = self.dropout3(e3_pool)  # (B, 64, L, 8, 8)
        
        # BOTTLENECK
        b = self.bottleneck_relu(self.bottleneck_conv(e3_pool))
        b = self.dropout_bottleneck(b)  # (B, 128, L, 8, 8)
        
        # SKIP CONNECTION: concatener encoder3 output avec bottleneck
        skip = torch.cat([e3_pool, b], dim=1)  # (B, 64+128=192, L, 8, 8)
        skip = self.skip_relu(self.skip_conv(skip))  # (B, 128, L, 8, 8)
        
        # GLOBAL POOLING
        pooled = self.global_pool(skip)  # (B, 128, 1, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # (B, 128)
        pooled = self.dropout_pool(pooled)
        
        # MLP HEAD
        h = self.fc1_relu(self.fc1(pooled))  # (B, 256)
        h = self.fc_dropout1(h)
        h = self.fc2_relu(self.fc2(h))  # (B, 128)
        h = self.fc_dropout2(h)
        out = self.fc_out(h)  # (B, 1)
        
        return out
