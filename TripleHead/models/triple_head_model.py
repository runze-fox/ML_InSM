# -*- coding: utf-8 -*-
"""
TripleHead PE Surrogate Model
Three-head independent upsampling architecture: Shared backbone + Three independent PE component decoders
"""
import torch
import torch.nn as nn


class TripleHeadPEModel(nn.Module):
    """
    Three-head plastic strain surrogate model
    
    Architecture Design:
    - Shared Backbone: Maps thickness T to a shared feature map (base_channels × 8 × 8)
    - Independent Heads: PE11, PE22, PE33 each have independent upsampling decoders
    
    Design Concept:
    - Geometric feature sharing (T-joints of the same thickness have identical geometry)
    - Physical mapping independence (Different strain components have different spatial distribution patterns)
    
    Args:
        base_channels: Number of channels for the shared feature map (default 128)
    """
    
    def __init__(self, base_channels=128):
        super(TripleHeadPEModel, self).__init__()
        self.base_channels = base_channels
        
        # Shared feature extractor: Thickness -> 8x8 feature map
        self.fc = nn.Sequential(
            nn.Linear(1, 512),
            nn.ReLU(True),
            nn.Linear(512, base_channels * 8 * 8),
            nn.ReLU(True)
        )
        
        # Three independent decoder heads
        self.head_pe11 = self._build_decoder(base_channels)
        self.head_pe22 = self._build_decoder(base_channels)
        self.head_pe33 = self._build_decoder(base_channels)
    
    def _build_decoder(self, in_channels):
        """
        Build a single decoder branch
        Upsampling path: 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x256
        
        Args:
            in_channels: Number of input channels
        
        Returns:
            nn.Sequential: Decoder network
        """
        return nn.Sequential(
            # 8x8 -> 16x16
            nn.ConvTranspose2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            
            # 128x128 -> 256x256 (Final output is single channel)
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input thickness [batch_size, 1]
        
        Returns:
            output: Three-component plastic strain predictions [batch_size, 3, 256, 256]
        """
        # Shared feature extraction
        feat = self.fc(x).view(-1, self.base_channels, 8, 8)
        
        # Three independent branches
        pe11 = self.head_pe11(feat)  # [B, 1, 256, 256]
        pe22 = self.head_pe22(feat)  # [B, 1, 256, 256]
        pe33 = self.head_pe33(feat)  # [B, 1, 256, 256]
        
        # Concatenate into a 3-channel output
        output = torch.cat([pe11, pe22, pe33], dim=1)  # [B, 3, 256, 256]
        
        return output
    
    def get_num_parameters(self):
        """Return the total number of parameters in the model"""
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Test model
    print("=== Testing TripleHeadPEModel ===")
    
    # Test different channel configurations
    for base_ch in [32, 64, 128]:
        print(f"\nTesting base_channels = {base_ch}")
        model = TripleHeadPEModel(base_channels=base_ch)
        
        # Forward pass test
        batch_size = 4
        x = torch.randn(batch_size, 1)  # Thickness input
        y = model(x)
        
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {y.shape}")
        print(f"  Total parameters: {model.get_num_parameters():,}")
        
        # Validate shapes
        assert y.shape == (batch_size, 3, 256, 256), \
            f"Incorrect output shape: Expected ({batch_size}, 3, 256, 256), Actual {y.shape}"
    
    print("\n✓ All tests passed!")
