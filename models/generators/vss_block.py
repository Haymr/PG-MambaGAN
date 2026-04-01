"""
PG-MambaGAN — VSS Block (Visual State Space Block)

Core building block of the VSS-U-Net generator.
Implements SS2D (2D Selective Scan) using the mamba-ssm CUDA kernel.

Architecture per block:
    Input (B, C, H, W)
      → LayerNorm
      → SS2D (4-way scan via Mamba SSM)
      → Residual
      → FFN (Conv1x1 → GELU → Conv1x1)
      → Residual
    Output (B, C, H, W)

References:
    - VMamba: Visual State Space Model (Liu et al., 2024)
    - Mamba: Linear-Time Sequence Modeling (Gu & Dao, 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Try importing mamba-ssm; provide fallback info if unavailable
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


# ======================================================================
# Layer Norm for NCHW tensors
# ======================================================================

class LayerNorm2d(nn.Module):
    """Layer normalization for (B, C, H, W) tensors."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → permute → norm → permute back
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = F.layer_norm(x, [x.size(-1)], self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)  # (B, C, H, W)


# ======================================================================
# SS2D: 2D Selective Scan
# ======================================================================

class SS2D(nn.Module):
    """
    2D Selective Scan module.
    
    Flattens a 2D feature map into 4 directional 1D sequences,
    processes each with the Mamba SSM kernel, then merges results.
    
    Scan directions:
        1. Continuous Sweep (Z-shaped/Snake) Raster
        2. Reverse Continuous Sweep Raster
        3. Continuous Sweep (Z-shaped/Snake) Column-wise
        4. Reverse Continuous Sweep Column-wise
    
    Args:
        dim: Number of input/output channels.
        d_state: SSM state dimension (default: 16).
        d_conv: Local convolution width (default: 4).
        expand: Expansion factor for inner dim (default: 2).
        dropout: Dropout rate (default: 0.0).
    """
    
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba-ssm is required for SS2D. "
                "Install: pip install mamba-ssm causal-conv1d"
            )
        
        self.dim = dim
        self.n_directions = 4
        
        # Input projection: dim → dim (shared across directions)
        self.in_proj = nn.Linear(dim, dim)
        
        # One Mamba SSM per scan direction
        self.mamba_blocks = nn.ModuleList([
            Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(self.n_directions)
        ])
        
        # Output merge: 4*dim → dim
        self.out_proj = nn.Linear(dim * self.n_directions, dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def _flatten_2d_to_sequences(
        self, x: torch.Tensor
    ) -> list:
        """
        Convert (B, C, H, W) tensor to 4 directional sequences.
        
        Returns:
            List of 4 tensors, each (B, L, C) where L = H*W
        """
        B, C, H, W = x.shape
        
        # (B, C, H, W) → (B, H, W, C) for sequence processing
        x_hwc = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        
        # Direction 1: Z-shaped / Snake Raster (flip odd rows)
        x_raster = x_hwc.clone()
        x_raster[:, 1::2, :, :] = torch.flip(x_raster[:, 1::2, :, :], dims=[2])
        seq_raster = x_raster.reshape(B, H * W, C)
        
        # Direction 2: Reverse Z-shaped Raster
        seq_reverse = torch.flip(seq_raster, dims=[1])
        
        # Direction 3: Z-shaped Column-wise (transpose then flip odd columns)
        x_col = x_hwc.permute(0, 2, 1, 3).clone()  # (B, W, H, C)
        x_col[:, 1::2, :, :] = torch.flip(x_col[:, 1::2, :, :], dims=[2])
        seq_column = x_col.reshape(B, H * W, C)
        
        # Direction 4: Reverse Z-shaped Column-wise
        seq_col_rev = torch.flip(seq_column, dims=[1])
        
        return [seq_raster, seq_reverse, seq_column, seq_col_rev]
    
    def _merge_sequences(
        self,
        outputs: list,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """
        Merge 4 directional SSM outputs back to (B, C, H, W).
        
        Each output is (B, L, C). We reverse the directional transforms,
        then concatenate and project.
        """
        B = outputs[0].shape[0]
        C = outputs[0].shape[2]
        
        # Reverse the scan transforms
        # Dir 1: Z-shaped raster — reverse row flips
        out_1 = outputs[0].reshape(B, H, W, C).clone()
        out_1[:, 1::2, :, :] = torch.flip(out_1[:, 1::2, :, :], dims=[2])
        out_1 = out_1.reshape(B, H * W, C)
        
        # Dir 2: Reverse Z-shaped raster — flip 1D back, then reverse row flips
        out_2 = torch.flip(outputs[1], dims=[1])
        out_2 = out_2.reshape(B, H, W, C).clone()
        out_2[:, 1::2, :, :] = torch.flip(out_2[:, 1::2, :, :], dims=[2])
        out_2 = out_2.reshape(B, H * W, C)
        
        # Dir 3: Z-shaped column-wise — reverse col flips, then transpose back
        out_3 = outputs[2].reshape(B, W, H, C).clone()
        out_3[:, 1::2, :, :] = torch.flip(out_3[:, 1::2, :, :], dims=[2])
        out_3 = out_3.permute(0, 2, 1, 3).reshape(B, H * W, C)
        
        # Dir 4: Reverse Z-shaped column-wise — flip 1D, reverse col flips, transpose back
        out_4 = torch.flip(outputs[3], dims=[1])
        out_4 = out_4.reshape(B, W, H, C).clone()
        out_4[:, 1::2, :, :] = torch.flip(out_4[:, 1::2, :, :], dims=[2])
        out_4 = out_4.permute(0, 2, 1, 3).reshape(B, H * W, C)
        
        # Concatenate all directions: (B, H*W, 4*C)
        merged = torch.cat([out_1, out_2, out_3, out_4], dim=-1)
        
        # Project back to dim: (B, H*W, C)
        merged = self.out_proj(merged)
        
        # Reshape to spatial: (B, H, W, C) → (B, C, H, W)
        return merged.reshape(B, H, W, C).permute(0, 3, 1, 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            Output tensor (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Flatten to 4 directional sequences
        sequences = self._flatten_2d_to_sequences(x)
        
        # Project input
        sequences = [self.in_proj(seq) for seq in sequences]
        
        # Process each direction with its Mamba block
        outputs = []
        for mamba_block, seq in zip(self.mamba_blocks, sequences):
            out = mamba_block(seq)  # (B, L, C) → (B, L, C)
            outputs.append(out)
        
        # Merge back to 2D
        result = self._merge_sequences(outputs, H, W)
        
        return self.dropout(result)


# ======================================================================
# Feed-Forward Network
# ======================================================================

class FFN(nn.Module):
    """Feed-forward network with Conv1x1 → GELU → Conv1x1."""
    
    def __init__(self, dim: int, expand: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = dim * expand
        self.fc1 = nn.Conv2d(dim, hidden, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, dim, 1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


# ======================================================================
# VSS Block
# ======================================================================

class VSSBlock(nn.Module):
    """
    Visual State Space Block.
    
    Architecture:
        x → LayerNorm → SS2D → + residual
          → LayerNorm → FFN  → + residual
    
    Args:
        dim: Number of channels.
        d_state: Mamba SSM state dimension.
        d_conv: Mamba local convolution width.
        ssm_expand: Mamba inner dimension expand factor.
        ffn_expand: FFN hidden dimension expand factor.
        drop_rate: Dropout rate.
        drop_path_rate: Stochastic depth rate.
    """
    
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        ssm_expand: int = 2,
        ffn_expand: int = 4,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        
        # SS2D branch
        self.norm1 = LayerNorm2d(dim)
        self.ss2d = SS2D(
            dim=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=ssm_expand,
            dropout=drop_rate,
        )
        
        # FFN branch
        self.norm2 = LayerNorm2d(dim)
        self.ffn = FFN(dim=dim, expand=ffn_expand, dropout=drop_rate)
        
        # Drop path (stochastic depth)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SS2D with residual
        x = x + self.drop_path(self.ss2d(self.norm1(x)))
        # FFN with residual
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


# ======================================================================
# Drop Path (Stochastic Depth)
# ======================================================================

class DropPath(nn.Module):
    """
    Stochastic depth per sample (when applied in residual blocks).
    Reference: Deep Networks with Stochastic Depth (Huang et al., 2016)
    """
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        
        return x / keep_prob * random_tensor


# ======================================================================
# Patch Merging (Downsample 2×)
# ======================================================================

class PatchMerging(nn.Module):
    """
    Merge 2×2 patches into a single token, doubling channels.
    
    (B, C, H, W) → (B, 2C, H/2, W/2)
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.norm = LayerNorm2d(4 * dim)
        self.reduction = nn.Conv2d(4 * dim, 2 * dim, 1, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Extract 2×2 patches
        x0 = x[:, :, 0::2, 0::2]  # (B, C, H/2, W/2)
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]
        
        # Concatenate: (B, 4C, H/2, W/2)
        x = torch.cat([x0, x1, x2, x3], dim=1)
        
        # Normalize and reduce: (B, 4C, H/2, W/2) → (B, 2C, H/2, W/2)
        x = self.norm(x)
        x = self.reduction(x)
        
        return x


# ======================================================================
# Patch Expanding (Upsample 2×)
# ======================================================================

class PatchExpanding(nn.Module):
    """
    Expand patches, halving channels and doubling spatial dims.
    
    (B, C, H, W) → (B, C/2, 2H, 2W)
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.expand = nn.Conv2d(dim, dim * 2, 1, bias=False)
        self.norm = LayerNorm2d(dim // 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Expand channels: (B, C, H, W) → (B, 2C, H, W)
        x = self.expand(x)
        
        # Pixel shuffle: (B, 2C, H, W) → (B, C/2, 2H, 2W)
        x = x.reshape(B, C // 2, 4, H, W)
        x = x.permute(0, 1, 3, 4, 2)  # (B, C/2, H, W, 4)
        x = x.reshape(B, C // 2, H, W, 2, 2)
        x = x.permute(0, 1, 2, 4, 3, 5)  # (B, C/2, H, 2, W, 2)
        x = x.reshape(B, C // 2, 2 * H, 2 * W)
        
        return self.norm(x)


# ======================================================================
# VSS Stage (multiple VSS blocks)
# ======================================================================

class VSSStage(nn.Module):
    """
    A stage of VSS blocks with optional downsampling/upsampling.
    
    Supports gradient checkpointing for VRAM savings.
    
    Args:
        dim: Channel dimension.
        depth: Number of VSS blocks in this stage.
        d_state: Mamba state dimension.
        d_conv: Mamba conv width.
        ssm_expand: Mamba expand factor.
        ffn_expand: FFN expand factor.
        drop_rate: Dropout rate.
        drop_path_rates: List of drop path rates for each block.
        use_checkpoint: Enable gradient checkpointing.
    """
    
    def __init__(
        self,
        dim: int,
        depth: int,
        d_state: int = 16,
        d_conv: int = 4,
        ssm_expand: int = 2,
        ffn_expand: int = 4,
        drop_rate: float = 0.0,
        drop_path_rates: Optional[list] = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        
        self.use_checkpoint = use_checkpoint
        
        if drop_path_rates is None:
            drop_path_rates = [0.0] * depth
        
        self.blocks = nn.ModuleList([
            VSSBlock(
                dim=dim,
                d_state=d_state,
                d_conv=d_conv,
                ssm_expand=ssm_expand,
                ffn_expand=ffn_expand,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rates[i],
            )
            for i in range(depth)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    blk, x, use_reentrant=False
                )
            else:
                x = blk(x)
        return x
