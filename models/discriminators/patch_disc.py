"""
PG-MambaGAN — SN-PatchGAN Discriminator (PyTorch)

PatchGAN discriminator with Spectral Normalization for WGAN-GP training.

Architecture:
    Input: (B, 2, 512, 512)  — concat(LDCT, NDCT_or_Denoised)
    → 4 conv blocks with increasing filters [64, 128, 256, 512]
    → Final conv → (B, 1, 62, 62) patch map
    → No sigmoid (WGAN-GP compatible)

Usage:
    disc = PatchDiscriminator(in_channels=2, use_spectral_norm=True)
    score = disc(torch.randn(1, 2, 512, 512))  # (1, 1, 62, 62)
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from typing import List


class PatchDiscriminator(nn.Module):
    """
    PatchGAN Discriminator with Spectral Normalization.
    
    Outputs a 2D map of real/fake scores for each receptive field patch.
    
    Args:
        in_channels: Number of input channels (2 = concat LDCT + target).
        base_filters: Base filter count (default: 64).
        n_layers: Number of discriminator layers (default: 4).
        use_spectral_norm: Apply spectral normalization (recommended for WGAN-GP).
        filters: Optional explicit filter list (overrides base_filters/n_layers).
    """
    
    def __init__(
        self,
        in_channels: int = 2,
        base_filters: int = 64,
        n_layers: int = 4,
        use_spectral_norm: bool = True,
        filters: List[int] = None,
    ):
        super().__init__()
        
        if filters is None:
            filters = [min(base_filters * (2 ** i), 512) for i in range(n_layers)]
            # Default: [64, 128, 256, 512]
        
        norm_fn = spectral_norm if use_spectral_norm else lambda x: x
        
        layers = []
        
        # First layer: no normalization
        layers.extend([
            norm_fn(nn.Conv2d(in_channels, filters[0], 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        ])
        
        # Middle layers
        prev_ch = filters[0]
        for i in range(1, len(filters)):
            curr_ch = filters[i]
            stride = 2 if i < len(filters) - 1 else 1  # Last middle layer: stride 1
            
            layers.extend([
                norm_fn(nn.Conv2d(prev_ch, curr_ch, 4, stride=stride, padding=1)),
                nn.InstanceNorm2d(curr_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            prev_ch = curr_ch
        
        # Final layer: 1-channel output (no activation for WGAN-GP)
        layers.append(
            norm_fn(nn.Conv2d(prev_ch, 1, 4, stride=1, padding=1))
        )
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Concatenated input (B, 2, H, W) where
               channel 0 = LDCT condition
               channel 1 = NDCT (real) or Denoised (fake)
        
        Returns:
            Patch score map (B, 1, H', W') — no sigmoid
        """
        return self.model(x)
    
    def get_param_count(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "total_M": f"{total / 1e6:.2f}M"}
