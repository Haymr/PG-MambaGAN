"""
PG-MambaGAN — VSS-U-Net Generator

Reference: 
    Inspired by VMamba (Liu et al., 2024) and Mamba-ND. 
    Core SSM logic adapted from `mamba-ssm` (Gu & Dao, 2023).

Full Visual State Space U-Net architecture for LDCT denoising.
Replaces CNN encoder/decoder with VSS (Mamba-based) stages.

Architecture (512×512 input):
    ┌─ Patch Embed ─────────────────────────────────────────┐
    │  Conv 3×3, s=2, p=1 → (B, 96, 256, 256)              │
    ├─ Encoder ─────────────────────────────────────────────┤
    │  Stage 1: 2× VSS, dim=96  (256×256) → skip_1         │
    │  ↓ PatchMerging                                        │
    │  Stage 2: 2× VSS, dim=192 (128×128) → skip_2         │
    │  ↓ PatchMerging                                        │
    │  Stage 3: 4× VSS, dim=384 (64×64)   → skip_3         │
    │  ↓ PatchMerging                                        │
    │  Stage 4: 2× VSS, dim=768 (32×32)   Bottleneck       │
    ├─ Decoder ─────────────────────────────────────────────┤
    │  ↑ PatchExpanding + skip_3 → 4× VSS, dim=384 (64×64) │
    │  ↑ PatchExpanding + skip_2 → 2× VSS, dim=192(128×128)│
    │  ↑ PatchExpanding + skip_1 → 2× VSS, dim=96 (256×256)│
    ├─ Head ────────────────────────────────────────────────┤
    │  Bilinear ×2 → Conv 3×3 → Conv 3×3 → tanh            │
    └───────────────────────────────────────────────────────┘

Usage:
    from models.generators.vss_unet import VSSUNet
    
    model = VSSUNet(
        in_channels=1,
        out_channels=1,
        embed_dim=96,
        depths=[2, 2, 4, 2],
        use_checkpoint=True,  # gradient checkpointing for VRAM
    )
    
    x = torch.randn(1, 1, 512, 512)
    y = model(x)  # (1, 1, 512, 512)
"""

import torch
import torch.nn as nn
from typing import List, Optional

from models.generators.vss_block import (
    VSSStage,
    PatchMerging,
    PatchExpanding,
    LayerNorm2d,
)


class VSSUNet(nn.Module):
    """
    VSS-U-Net Generator for LDCT Denoising.
    
    Full Mamba-based encoder-decoder with skip connections.
    
    Args:
        in_channels: Input channels (1 for CT).
        out_channels: Output channels (1 for CT).
        embed_dim: Base embedding dimension (default: 96).
        depths: Number of VSS blocks per stage [enc1, enc2, enc3, bottleneck].
        d_state: Mamba SSM state dimension.
        d_conv: Mamba local convolution width.
        ssm_expand: Mamba inner expand factor.
        ffn_expand: FFN hidden expand factor.
        patch_size: Initial patch embedding size (default: 4).
        drop_rate: Dropout rate.
        drop_path_rate: Max stochastic depth rate (linearly increases).
        use_checkpoint: Enable gradient checkpointing (saves ~60% VRAM).
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 4, 2],
        d_state: int = 16,
        d_conv: int = 4,
        ssm_expand: int = 2,
        ffn_expand: int = 4,
        patch_size: int = 4,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        
        self.num_stages = len(depths)
        assert self.num_stages == 4, "VSS-U-Net requires exactly 4 encoder stages"
        
        # Channel dims at each stage
        dims = [embed_dim * (2 ** i) for i in range(self.num_stages)]
        # e.g., [96, 192, 384, 768]
        
        # Stochastic depth rates (linearly increasing)
        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        
        # ==============================================================
        # Patch Embedding
        # ==============================================================
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, 
                      kernel_size=3 if patch_size < 4 else patch_size,
                      stride=patch_size, 
                      padding=1 if patch_size < 4 else 0),
            LayerNorm2d(embed_dim),
        )
        # 512×512 → 256×256
        
        # ==============================================================
        # Encoder
        # ==============================================================
        self.encoder_stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        block_idx = 0
        for i in range(self.num_stages):
            stage = VSSStage(
                dim=dims[i],
                depth=depths[i],
                d_state=d_state,
                d_conv=d_conv,
                ssm_expand=ssm_expand,
                ffn_expand=ffn_expand,
                drop_rate=drop_rate,
                drop_path_rates=dpr[block_idx:block_idx + depths[i]],
                use_checkpoint=use_checkpoint,
            )
            self.encoder_stages.append(stage)
            block_idx += depths[i]
            
            # Downsample between encoder stages (not after the last one)
            if i < self.num_stages - 1:
                self.downsample_layers.append(PatchMerging(dims[i]))
        
        # ==============================================================
        # Decoder
        # ==============================================================
        # Decoder is symmetric: stages 3→2→1 (no stage for bottleneck)
        decoder_depths = list(reversed(depths[:-1]))  # [4, 2, 2]
        decoder_dims = list(reversed(dims[:-1]))       # [384, 192, 96]
        
        self.upsample_layers = nn.ModuleList()
        self.skip_proj = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()
        
        for i in range(self.num_stages - 1):
            # Upsample from previous decoder/bottleneck dim
            up_in_dim = dims[self.num_stages - 1 - i]
            self.upsample_layers.append(PatchExpanding(up_in_dim))
            
            # Skip connection projection: concat → reduce channels
            # After upsample: dim/2, skip: decoder_dims[i], concat: dim/2 + decoder_dims[i]
            skip_cat_dim = up_in_dim // 2 + decoder_dims[i]
            self.skip_proj.append(
                nn.Sequential(
                    nn.Conv2d(skip_cat_dim, decoder_dims[i], 1, bias=False),
                    LayerNorm2d(decoder_dims[i]),
                )
            )
            
            # Decoder stage (reuses drop path rates from encoder symmetrically)
            self.decoder_stages.append(
                VSSStage(
                    dim=decoder_dims[i],
                    depth=decoder_depths[i],
                    d_state=d_state,
                    d_conv=d_conv,
                    ssm_expand=ssm_expand,
                    ffn_expand=ffn_expand,
                    drop_rate=drop_rate,
                    drop_path_rates=[0.0] * decoder_depths[i],
                    use_checkpoint=use_checkpoint,
                )
            )
        
        self.final_upsample = nn.Sequential(
            # 1. Eksiksiz yüksek çözünürlüğe taşı (Yüksek Kalite)
            nn.Upsample(scale_factor=patch_size, mode='bilinear', align_corners=False),
            
            # 2. Receptive field'ı genişlet ve kanalları düşür
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            LayerNorm2d(embed_dim // 2),
            nn.GELU(),
            
            # 3. Artefaktları tamamen yok eden (smoothing) 2. Evrişim
            nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3, padding=1),
            LayerNorm2d(embed_dim // 2),
            nn.GELU(),
            
            # 4. Çıktı
            nn.Conv2d(embed_dim // 2, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Weight initialization."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, LayerNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input LDCT tensor (B, 1, 512, 512)
            
        Returns:
            Denoised output tensor (B, 1, 512, 512)
        """
        # Patch embedding: (B, 1, 512, 512) → (B, 96, 256, 256)
        x = self.patch_embed(x)
        
        # ----- Encoder -----
        skips = []
        for i, stage in enumerate(self.encoder_stages):
            x = stage(x)
            
            if i < self.num_stages - 1:
                skips.append(x)  # Save skip before downsampling
                x = self.downsample_layers[i](x)
        
        # x is now the bottleneck: (B, 768, 32, 32)
        
        # ----- Decoder -----
        for i in range(self.num_stages - 1):
            # Upsample
            x = self.upsample_layers[i](x)
            
            # Skip connection (reverse order)
            skip = skips[self.num_stages - 2 - i]
            x = torch.cat([x, skip], dim=1)
            x = self.skip_proj[i](x)
            
            # Decoder VSS stage
            x = self.decoder_stages[i](x)
        
        # x is now (B, 96, 256, 256)
        
        # ----- Final upsample -----
        # (B, 96, 256, 256) → (B, 1, 512, 512)
        x = self.final_upsample(x)
        
        return x
    
    def get_param_count(self) -> dict:
        """Return parameter count summary."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total": total,
            "trainable": trainable,
            "total_M": f"{total / 1e6:.2f}M",
            "trainable_M": f"{trainable / 1e6:.2f}M",
        }
