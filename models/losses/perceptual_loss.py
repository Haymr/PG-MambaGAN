"""
PG-MambaGAN — Perceptual Loss (VGG19 + LPIPS) (PyTorch)

Feature-level loss using pretrained VGG19 intermediate layers.
Optional LPIPS (Learned Perceptual Image Patch Similarity) integration.

Note: CT images are single-channel. We replicate to 3 channels for VGG.

Usage:
    criterion = PerceptualLoss(use_lpips=True)
    loss = criterion(predicted, ndct)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

import torchvision.models as models


class PerceptualLoss(nn.Module):
    """
    VGG19-based perceptual loss + optional LPIPS.
    
    Extracts features from intermediate VGG19 layers and computes
    L1/L2 distance between predicted and target feature maps.
    
    Args:
        layer_indices: VGG19 feature layer indices to extract.
            Default: [3, 8, 17, 26] = relu1_2, relu2_2, relu3_4, relu4_4
        layer_weights: Weight for each layer's contribution.
        loss_type: "l1" or "l2".
        use_lpips: Also compute LPIPS distance (requires 'lpips' package).
        lpips_weight: Weight for LPIPS relative to VGG loss.
    """
    
    def __init__(
        self,
        layer_indices: List[int] = [3, 8, 17, 26],
        layer_weights: Optional[List[float]] = None,
        loss_type: str = "l1",
        use_lpips: bool = False,
        lpips_weight: float = 1.0,
    ):
        super().__init__()
        
        self.layer_indices = layer_indices
        self.layer_weights = layer_weights or [1.0] * len(layer_indices)
        self.loss_type = loss_type
        self.lpips_weight = lpips_weight
        
        # Load pretrained VGG19 features
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        
        # Freeze all VGG weights
        for param in vgg.parameters():
            param.requires_grad = False
        
        # Split VGG into chunks for intermediate feature extraction
        self.vgg_slices = nn.ModuleList()
        prev_idx = 0
        for idx in sorted(layer_indices):
            self.vgg_slices.append(vgg[prev_idx:idx + 1])
            prev_idx = idx + 1
        
        # LPIPS (optional)
        self.use_lpips = use_lpips
        self.lpips_fn = None
        if use_lpips:
            try:
                import lpips
                self.lpips_fn = lpips.LPIPS(net="vgg", verbose=False)
                self.lpips_fn.eval()
                for param in self.lpips_fn.parameters():
                    param.requires_grad = False
            except ImportError:
                print("⚠️  lpips not installed. LPIPS will be skipped.")
                self.use_lpips = False
        
        # ImageNet normalization constants
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
    
    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert single-channel CT to 3-channel VGG-compatible input.
        
        Steps:
            1. CT [-1, 1] → [0, 1]
            2. Replicate to 3 channels
            3. Normalize with ImageNet stats
        """
        # [-1, 1] → [0, 1]
        x = (x + 1.0) / 2.0
        
        # 1 channel → 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # ImageNet normalize
        x = (x - self.mean) / self.std
        
        return x
    
    def _extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract intermediate features from VGG slices."""
        features = []
        for slice_module in self.vgg_slices:
            x = slice_module(x)
            features.append(x)
        return features
    
    def forward(
        self, predicted: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute perceptual loss.
        
        Args:
            predicted: Generator output (B, 1, H, W), [-1, 1].
            target: NDCT ground truth (B, 1, H, W), [-1, 1].
            
        Returns:
            Scalar perceptual loss.
        """
        # Prepare VGG inputs
        pred_vgg = self._prepare_input(predicted)
        tgt_vgg = self._prepare_input(target)
        
        # Extract features
        pred_features = self._extract_features(pred_vgg)
        tgt_features = self._extract_features(tgt_vgg)
        
        # Compute weighted feature loss
        total_loss = torch.tensor(0.0, device=predicted.device)
        
        for feat_p, feat_t, w in zip(pred_features, tgt_features, self.layer_weights):
            if self.loss_type == "l1":
                total_loss = total_loss + w * F.l1_loss(feat_p, feat_t)
            else:
                total_loss = total_loss + w * F.mse_loss(feat_p, feat_t)
        
        # LPIPS (optional)
        if self.use_lpips and self.lpips_fn is not None:
            with torch.no_grad():
                lpips_val = self.lpips_fn(pred_vgg, tgt_vgg).mean()
            total_loss = total_loss + self.lpips_weight * lpips_val
        
        return total_loss
