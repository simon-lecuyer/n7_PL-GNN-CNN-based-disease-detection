import torch
import torch.nn as nn


class DiseaseCNN(nn.Module):
    """
    3D CNN pour prédiction temporelle de l'intensité d'infection.
    Input: (B, L, 64, 64)  - L frames temporelles
    Output: (B, 1, 64, 64) - prédiction au temps t+L+1
    """
    
    def __init__(self, in_frames=5, out_channels=1):
        """
        Args:
            in_frames: nombre de frames temporelles en entrée (L)
            out_channels: nombre de canaux de sortie (1 pour l'intensité)
        """
        super(DiseaseCNN, self).__init__()
        
        # ───── ENCODER 3D ─────
        # Block 1: (B, L, 64, 64) → (B, 32, L, 32, 32)
        self.enc1_conv = nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.enc1_relu = nn.ReLU(inplace=True)
        self.enc1_pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        # Block 2: (B, 32, L, 32, 32) → (B, 64, L, 16, 16)
        self.enc2_conv = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.enc2_relu = nn.ReLU(inplace=True)
        self.enc2_pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        # Block 3: (B, 64, L, 16, 16) → (B, 128, L, 8, 8)
        self.enc3_conv = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.enc3_relu = nn.ReLU(inplace=True)
        self.enc3_pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        # ───── BOTTLENECK 3D ─────
        self.bottleneck_conv1 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bottleneck_relu1 = nn.ReLU(inplace=True)
        self.bottleneck_conv2 = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bottleneck_relu2 = nn.ReLU(inplace=True)
        
        # ───── TEMPORAL COMPRESSION ─────
        # Réduire la dimension temporelle: (B, 256, L, 8, 8) → (B, 256, 1, 8, 8)
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, 8, 8))
        
        # ───── DECODER 2D ─────
        # De là, on passe à des convolutions 2D standard
        # (B, 256, 1, 8, 8) → squeeze → (B, 256, 8, 8)
        
        # Block 1: 8 → 16
        self.dec1_upconv = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1_conv = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.dec1_relu = nn.ReLU(inplace=True)
        
        # Block 2: 16 → 32
        self.dec2_upconv = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dec2_relu = nn.ReLU(inplace=True)
        
        # Block 3: 32 → 64
        self.dec3_upconv = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec3_conv = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.dec3_relu = nn.ReLU(inplace=True)
        
        # Output
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Forward pass
        Args:
            x: (B, L, 64, 64) - L frames temporelles
        Returns:
            out: (B, 1, 64, 64) - prédiction seg
        """
        # Ajouter dimension channel pour Conv3d: (B, L, 64, 64) → (B, 1, L, 64, 64)
        x = x.unsqueeze(1)
        
        #  ENCODER 3D 
        e1 = self.enc1_relu(self.enc1_conv(x))
        e1_pool = self.enc1_pool(e1)
        
        e2 = self.enc2_relu(self.enc2_conv(e1_pool))
        e2_pool = self.enc2_pool(e2)
        
        e3 = self.enc3_relu(self.enc3_conv(e2_pool))
        e3_pool = self.enc3_pool(e3)
        
        # BOTTLENECK 
        b = self.bottleneck_relu2(self.bottleneck_conv2(self.bottleneck_relu1(self.bottleneck_conv1(e3_pool))))
        
        # TEMPORAL COMPRESSION
        b = self.temporal_pool(b)  # (B, 256, 1, 8, 8)
        b = b.squeeze(2)  # (B, 256, 8, 8)
        
        # DECODER 2D 
        d1 = self.dec1_upconv(b)
        d1 = self.dec1_relu(self.dec1_conv(d1))
        
        d2 = self.dec2_upconv(d1)
        d2 = self.dec2_relu(self.dec2_conv(d2))
        
        d3 = self.dec3_upconv(d2)
        d3 = self.dec3_relu(self.dec3_conv(d3))
        
        # Output
        out = self.final_conv(d3)
        out = self.sigmoid(out)
        
        return out
