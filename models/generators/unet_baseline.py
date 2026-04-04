"""
PG-MambaGAN — Baseline U-Net Generator

Reference:
    Baseline U-Net adapted from basic Medical Image Denoising conventions.

Standard CNN-based U-Net architecture for baseline ablation studies.
This is the PyTorch equivalent of the original TF-based generator.

Architecture:
    8 encoder blocks → 7 decoder blocks with skip connections
    All Conv2d with LeakyReLU + InstanceNorm

Usage:
    model = UNetBaseline(in_channels=1, out_channels=1)
    y = model(torch.randn(1, 1, 512, 512))  # (1, 1, 512, 512)
"""

import torch
import torch.nn as nn
from typing import List


class ConvBlock(nn.Module):
    """Conv2d → InstanceNorm → LeakyReLU."""
    
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        use_norm: bool = True,
        use_act: bool = True,
    ):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=not use_norm)]
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_ch))
        if use_act:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpConvBlock(nn.Module):
    """ConvTranspose2d → InstanceNorm → ReLU + optional Dropout."""
    
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        use_dropout: bool = False,
    ):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetBaseline(nn.Module):
    """
    Standard U-Net Generator (Pix2Pix style).
    
    Encoder: 8 downsampling blocks
    Decoder: 7 upsampling blocks with skip connections
    
    Args:
        in_channels: Input channels (1 for CT).
        out_channels: Output channels (1 for CT).
        base_filters: Base filter count (default: 64).
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_filters: int = 64,
    ):
        super().__init__()
        
        bf = base_filters
        
        # Encoder (8 blocks): 512→256→128→64→32→16→8→4→2
        self.enc1 = ConvBlock(in_channels, bf, use_norm=False)    # 256
        self.enc2 = ConvBlock(bf, bf * 2)                          # 128
        self.enc3 = ConvBlock(bf * 2, bf * 4)                      # 64
        self.enc4 = ConvBlock(bf * 4, bf * 8)                      # 32
        self.enc5 = ConvBlock(bf * 8, bf * 8)                      # 16
        self.enc6 = ConvBlock(bf * 8, bf * 8)                      # 8
        self.enc7 = ConvBlock(bf * 8, bf * 8)                      # 4
        self.enc8 = ConvBlock(bf * 8, bf * 8, use_norm=False)      # 2
        
        # Decoder (7 blocks, concat skip connections)
        self.dec7 = UpConvBlock(bf * 8, bf * 8, use_dropout=True)      # 4
        self.dec6 = UpConvBlock(bf * 16, bf * 8, use_dropout=True)     # 8
        self.dec5 = UpConvBlock(bf * 16, bf * 8, use_dropout=True)     # 16
        self.dec4 = UpConvBlock(bf * 16, bf * 8)                        # 32
        self.dec3 = UpConvBlock(bf * 16, bf * 4)                        # 64
        self.dec2 = UpConvBlock(bf * 8, bf * 2)                         # 128
        self.dec1 = UpConvBlock(bf * 4, bf)                             # 256
        
        # Final output: 256→512
        self.final = nn.Sequential(
            nn.ConvTranspose2d(bf * 2, out_channels, 4, 2, 1),
            nn.Tanh(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)
        
        # Decoder with skip connections
        d7 = self.dec7(e8)
        d7 = torch.cat([d7, e7], dim=1)
        
        d6 = self.dec6(d7)
        d6 = torch.cat([d6, e6], dim=1)
        
        d5 = self.dec5(d6)
        d5 = torch.cat([d5, e5], dim=1)
        
        d4 = self.dec4(d5)
        d4 = torch.cat([d4, e4], dim=1)
        
        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        
        d1 = self.dec1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        
        return self.final(d1)
    
    def get_param_count(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "total_M": f"{total / 1e6:.2f}M"}
