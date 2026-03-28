"""
PG-MambaGAN — Multi-Scale FFT Frequency Loss (PyTorch)

Penalizes frequency domain differences between predicted and target images.
Uses 2D FFT at multiple scales to capture both global and local frequency info.

Usage:
    criterion = FrequencyLoss(scales=[1.0, 0.5, 0.25])
    loss = criterion(predicted, ndct)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class FrequencyLoss(nn.Module):
    """
    Multi-scale FFT-based frequency loss.
    
    Computes L1 difference between frequency spectra of predicted and
    target images at multiple spatial scales.
    
    Args:
        scales: List of scale factors for multi-scale computation.
                1.0 = original resolution, 0.5 = half resolution, etc.
        loss_type: "l1" or "l2" for spectrum comparison.
        log_scale: If True, compares log-magnitude spectra (more perceptually
                   meaningful, better gradient properties).
    """
    
    def __init__(
        self,
        scales: List[float] = [1.0, 0.5, 0.25],
        loss_type: str = "l1",
        log_scale: bool = True,
    ):
        super().__init__()
        
        self.scales = scales
        self.loss_type = loss_type
        self.log_scale = log_scale
    
    def _compute_fft_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """Compute centered FFT magnitude spectrum."""
        fft = torch.fft.fft2(x)
        fft_shifted = torch.fft.fftshift(fft)
        magnitude = torch.abs(fft_shifted)
        
        if self.log_scale:
            magnitude = torch.log1p(magnitude)  # log(1 + |FFT|)
        
        return magnitude
    
    def forward(
        self, predicted: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute multi-scale frequency loss.
        
        Args:
            predicted: Generator output (B, 1, H, W).
            target: Ground truth NDCT (B, 1, H, W).
            
        Returns:
            Scalar frequency loss.
        """
        total_loss = torch.tensor(0.0, device=predicted.device)
        
        for scale in self.scales:
            if scale < 1.0:
                # Downsample to target scale
                h = int(predicted.shape[2] * scale)
                w = int(predicted.shape[3] * scale)
                pred_scaled = F.interpolate(
                    predicted, size=(h, w), mode="bilinear", align_corners=False
                )
                tgt_scaled = F.interpolate(
                    target, size=(h, w), mode="bilinear", align_corners=False
                )
            else:
                pred_scaled = predicted
                tgt_scaled = target
            
            # Compute FFT magnitudes
            mag_pred = self._compute_fft_magnitude(pred_scaled)
            mag_tgt = self._compute_fft_magnitude(tgt_scaled)
            
            # Spectral difference
            if self.loss_type == "l1":
                scale_loss = F.l1_loss(mag_pred, mag_tgt)
            else:
                scale_loss = F.mse_loss(mag_pred, mag_tgt)
            
            total_loss = total_loss + scale_loss
        
        return total_loss / len(self.scales)
