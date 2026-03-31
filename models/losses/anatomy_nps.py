"""
PG-MambaGAN — Anatomy-Aware NPS Loss (Novelty #2)

Physics-guided loss function that computes Noise Power Spectrum (NPS)
separately for each anatomical tissue type, weighted by clinical importance.

██ CRITICAL DESIGN RULES ██

1. MASKING FROM GROUND TRUTH ONLY:
   Tissue masks are computed EXCLUSIVELY from NDCT (normal-dose ground truth).
   NEVER from LDCT or predicted images — quantum noise in LDCT causes massive
   HU deviations that would poison the tissue classification.

2. GRADIENT FLOW PROTECTION:
   Mask tensors are computed under @torch.no_grad() and .detach()ed from
   the computational graph. The masks guide WHERE to compute NPS, but
   they must NOT interfere with the generator's backpropagation.

3. MORPHOLOGICAL CLEANUP:
   After HU thresholding, binary closing/opening removes isolated noise
   pixels and fills small holes. Implemented via max_pool2d (pure PyTorch,
   no external dependencies, fully CUDA-compatible).

Pipeline:
    NDCT ──→ HU denormalize ──→ HU thresholding ──→ morphological clean
         ──→ tissue masks (detached) ──→ apply to BOTH ndct & predicted
         ──→ per-tissue NPS (2D FFT) ──→ radial average ──→ weighted loss

Usage:
    criterion = AnatomyAwareNPSLoss(
        hu_min=-1000, hu_max=1000,
        tissue_weights={"soft_tissue": 2.0, "lung": 1.5, ...},
    )
    loss = criterion(predicted, ndct_target)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# ======================================================================
# Tissue HU Thresholds
# ======================================================================

TISSUE_THRESHOLDS = {
    # tissue_name: (hu_min, hu_max)
    "air":         (-1000, -900),
    "lung":        (-900,  -500),
    "fat":         (-500,  -100),
    "soft_tissue": (-100,   300),
    "bone":        ( 300,  1000),
}

DEFAULT_TISSUE_WEIGHTS = {
    "air":         0.0,   # Ignored — no clinical relevance
    "lung":        1.5,   # Important for nodule detection
    "fat":         1.0,   # Standard
    "soft_tissue": 2.0,   # Highest priority — organs, tumors
    "bone":        0.5,   # Lower priority
}


# ======================================================================
# Morphological Operations (Pure PyTorch)
# ======================================================================

def _dilate(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Binary dilation via max_pool2d."""
    pad = kernel_size // 2
    return F.max_pool2d(
        mask.float(), kernel_size=kernel_size, stride=1, padding=pad
    )


def _erode(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Binary erosion via negated max_pool on negated input."""
    pad = kernel_size // 2
    return -F.max_pool2d(
        -mask.float(), kernel_size=kernel_size, stride=1, padding=pad
    )


def morphological_close(mask: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """
    Binary closing: dilation → erosion.
    Fills small holes in the mask.
    """
    return _erode(_dilate(mask, kernel_size), kernel_size)


def morphological_open(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    Binary opening: erosion → dilation.
    Removes isolated noise pixels from the mask.
    """
    return _dilate(_erode(mask, kernel_size), kernel_size)


def clean_mask(
    mask: torch.Tensor,
    close_kernel: int = 5,
    open_kernel: int = 3,
    min_area: int = 4096,  # 64² pixels
) -> torch.Tensor:
    """
    Full morphological cleanup pipeline.
    
    Steps:
        1. Binary closing (kernel=5×5) → fill small holes
        2. Binary opening (kernel=3×3) → remove isolated pixels
        3. Min area filter → discard tiny regions
    
    Args:
        mask: Binary mask tensor (B, 1, H, W), values in {0, 1}.
        close_kernel: Closing kernel size.
        open_kernel: Opening kernel size.
        min_area: Minimum region area in pixels. Regions smaller than
                  this are removed.
    
    Returns:
        Cleaned binary mask (B, 1, H, W).
    """
    # Step 1: Closing — fill holes
    mask = morphological_close(mask, close_kernel)
    
    # Step 2: Opening — remove noise
    mask = morphological_open(mask, open_kernel)
    
    # Step 3: Min area filter — remove tiny regions
    # Compute area per sample
    for b in range(mask.shape[0]):
        area = mask[b].sum().item()
        if area < min_area:
            mask[b] = 0.0
    
    # Re-binarize (pooling can produce soft values)
    mask = (mask > 0.5).float()
    
    return mask


# ======================================================================
# Anatomy-Aware NPS Loss
# ======================================================================

class AnatomyAwareNPSLoss(nn.Module):
    """
    Anatomy-Aware Noise Power Spectrum Loss.
    
    Computes NPS separately for each tissue type using masks derived
    from the NDCT ground truth image. The per-tissue NPS differences
    are weighted by clinical importance.
    
    ██ RULE 1: Masks come from NDCT ONLY ██
    ██ RULE 2: Masks are detached from grad graph ██
    ██ RULE 3: Morphological cleanup is mandatory ██
    
    Args:
        hu_min: Minimum HU value used during normalization (default: -1000).
        hu_max: Maximum HU value used during normalization (default: 1000).
        tissue_weights: Dict of tissue_name → weight for loss computation.
        patch_size: NPS computation patch size (default: 64).
        n_patches: Number of random patches per tissue region (default: 8).
        close_kernel: Morphological closing kernel size (default: 5).
        open_kernel: Morphological opening kernel size (default: 3).
        min_area: Minimum tissue region area in pixels (default: 64²).
    """
    
    def __init__(
        self,
        hu_min: int = -1000,
        hu_max: int = 1000,
        tissue_weights: Optional[Dict[str, float]] = None,
        patch_size: int = 64,
        n_patches: int = 8,
        close_kernel: int = 5,
        open_kernel: int = 3,
        min_area: int = 4096,
    ):
        super().__init__()
        
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.tissue_weights = tissue_weights or DEFAULT_TISSUE_WEIGHTS
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.close_kernel = close_kernel
        self.open_kernel = open_kernel
        self.min_area = min_area
    
    # ------------------------------------------------------------------
    # Step 1: Denormalize to HU space
    # ------------------------------------------------------------------
    
    def _denormalize_to_hu(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert normalized [-1, 1] tensor back to HU values.
        
        Inverse of: norm = (hu - hu_min) / (hu_max - hu_min) * 2 - 1
        Therefore:  hu = (norm + 1) / 2 * (hu_max - hu_min) + hu_min
        """
        return torch.clamp((x + 1.0) / 2.0 * (self.hu_max - self.hu_min) + self.hu_min, min=self.hu_min, max=self.hu_max)
    
    # ------------------------------------------------------------------
    # Step 2-3: Create tissue masks from NDCT (DETACHED)
    # ------------------------------------------------------------------
    
    @torch.no_grad()
    def _create_tissue_masks(
        self, ndct: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Create tissue segmentation masks from NDCT ground truth.
        
        ██ CRITICAL: This operates ONLY on NDCT ██
        ██ CRITICAL: All outputs are detached from grad graph ██
        
        Args:
            ndct: NDCT tensor in normalized [-1, 1] range, shape (B, 1, H, W).
            
        Returns:
            Dict mapping tissue_name → binary mask (B, 1, H, W), detached.
        """
        # Convert to HU space for thresholding
        ndct_hu = self._denormalize_to_hu(ndct)
        
        masks = {}
        
        for tissue_name, (t_min, t_max) in TISSUE_THRESHOLDS.items():
            # Skip tissues with zero weight
            if self.tissue_weights.get(tissue_name, 0.0) == 0.0:
                continue
            
            # HU thresholding on NDCT
            mask = ((ndct_hu >= t_min) & (ndct_hu < t_max)).float()
            
            # Morphological cleanup
            mask = clean_mask(
                mask,
                close_kernel=self.close_kernel,
                open_kernel=self.open_kernel,
                min_area=self.min_area,
            )
            
            # ██ DETACH from computational graph ██
            masks[tissue_name] = mask.detach()
        
        return masks
    
    # ------------------------------------------------------------------
    # Step 4: Extract masked patches
    # ------------------------------------------------------------------
    
    def _extract_paired_patches(
        self,
        pred_image: torch.Tensor,
        ndct_image: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        B, _, H, W = pred_image.shape
        ps = self.patch_size

        all_pred_patches = []
        all_ndct_patches = []

        for b in range(B):
            pred_b = pred_image[b, 0]
            ndct_b = ndct_image[b, 0]
            mask_b = mask[b, 0]

            if mask_b.sum() < ps * ps:
                continue

            valid_positions = []
            stride = ps // 2
            for y in range(0, H - ps + 1, stride):
                for x in range(0, W - ps + 1, stride):
                    patch_mask = mask_b[y:y + ps, x:x + ps]
                    if patch_mask.mean().item() >= 0.75:
                        valid_positions.append((y, x))

            if not valid_positions:
                continue

            n = min(self.n_patches, len(valid_positions))
            indices = torch.randperm(len(valid_positions))[:n]

            for idx in indices:
                y, x = valid_positions[idx]
                all_pred_patches.append(pred_b[y:y + ps, x:x + ps])
                all_ndct_patches.append(ndct_b[y:y + ps, x:x + ps])

        if not all_pred_patches:
            return None, None

        return torch.stack(all_pred_patches), torch.stack(all_ndct_patches)

    def _extract_masked_patches(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Extract patches from image within the masked region.
        
        Args:
            image: Image tensor (B, 1, H, W).
            mask: Binary mask (B, 1, H, W), detached.
            
        Returns:
            Patches tensor (N, patch_size, patch_size) or None if
            insufficient valid region.
        """
        B, _, H, W = image.shape
        ps = self.patch_size
        
        all_patches = []
        
        for b in range(B):
            # Find valid patch positions (where mask has enough coverage)
            img_b = image[b, 0]   # (H, W)
            mask_b = mask[b, 0]   # (H, W)
            
            if mask_b.sum() < ps * ps:
                continue
            
            # Grid of possible patch top-left corners
            valid_positions = []
            
            # Stride to avoid too many candidates
            stride = ps // 2
            for y in range(0, H - ps + 1, stride):
                for x in range(0, W - ps + 1, stride):
                    patch_mask = mask_b[y:y + ps, x:x + ps]
                    # Require at least 75% coverage within the patch
                    coverage = patch_mask.mean().item()
                    if coverage >= 0.75:
                        valid_positions.append((y, x))
            
            if not valid_positions:
                continue
            
            # Random sample from valid positions
            n = min(self.n_patches, len(valid_positions))
            indices = torch.randperm(len(valid_positions))[:n]
            
            for idx in indices:
                y, x = valid_positions[idx]
                patch = img_b[y:y + ps, x:x + ps]
                all_patches.append(patch)
        
        if not all_patches:
            return None
        
        return torch.stack(all_patches)  # (N, ps, ps)
    
    # ------------------------------------------------------------------
    # Step 5: Compute NPS via 2D FFT
    # ------------------------------------------------------------------
    
    def _compute_nps(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Compute 1D radially-averaged Noise Power Spectrum.
        
        Args:
            patches: Tensor (N, ps, ps).
            
        Returns:
            1D NPS profile (n_bins,) — radially averaged.
        """
        N, ps, _ = patches.shape
        
        # 2D FFT
        fft = torch.fft.fft2(patches)
        fft_shifted = torch.fft.fftshift(fft)
        
        # Power spectrum: |FFT|²
        power = (fft_shifted.real ** 2 + fft_shifted.imag ** 2) / (ps * ps)
        power = torch.log1p(power)

        center = ps // 2
        power[:, center, center] = 0.0
        
        # Average over all patches
        mean_power = power.mean(dim=0)  # (ps, ps)
        
        # Radial average
        return self._radial_average(mean_power)
    
    def _radial_average(self, power_2d: torch.Tensor) -> torch.Tensor:
        """Convert 2D power spectrum to 1D radial profile."""
        ps = power_2d.shape[0]
        center = ps // 2
        
        # Create distance matrix from center
        y, x = torch.meshgrid(
            torch.arange(ps, device=power_2d.device),
            torch.arange(ps, device=power_2d.device),
            indexing="ij",
        )
        dist = torch.sqrt((x - center).float() ** 2 + (y - center).float() ** 2)
        
        # Bin by integer distance
        max_radius = center
        n_bins = max_radius
        
        radial_profile = torch.zeros(n_bins, device=power_2d.device)
        bin_counts = torch.zeros(n_bins, device=power_2d.device)
        
        for r in range(n_bins):
            ring = (dist >= r) & (dist < r + 1)
            if ring.any():
                radial_profile[r] = power_2d[ring].mean()
                bin_counts[r] = ring.sum().float()
        
        # Normalize by non-zero bins
        valid = bin_counts > 0
        if valid.any():
            radial_profile[~valid] = 0.0
        
        return radial_profile
    
    # ------------------------------------------------------------------
    # Forward: Full NPS Loss Computation
    # ------------------------------------------------------------------
    
    def forward(
        self,
        predicted: torch.Tensor,
        ndct: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute Anatomy-Aware NPS Loss.
        
        Args:
            predicted: Generator output (B, 1, H, W), in [-1, 1].
            ndct: Normal-dose ground truth (B, 1, H, W), in [-1, 1].
            
        Returns:
            Tuple of:
                - Total weighted NPS loss (scalar tensor, differentiable).
                - Dict of per-tissue NPS losses (for logging).
        """
        import torchvision.transforms.functional as TF

        # Differentiable residual noise extraction on full tensors
        pred_noise_full = predicted - TF.gaussian_blur(predicted, kernel_size=5, sigma=1.0)
        ndct_noise_full = ndct - TF.gaussian_blur(ndct, kernel_size=5, sigma=1.0)

        # ██ Step 1-3: Create tissue masks from NDCT ONLY (detached) ██
        tissue_masks = self._create_tissue_masks(ndct)
        
        total_loss = torch.tensor(0.0, device=predicted.device, requires_grad=True)
        tissue_losses = {}
        n_valid = 0
        
        for tissue_name, mask in tissue_masks.items():
            weight = self.tissue_weights.get(tissue_name, 0.0)
            if weight == 0.0:
                continue
            
            # Apply SAME mask to BOTH ndct and predicted simultaneously
            pred_patches, ndct_patches = self._extract_paired_patches(
                pred_noise_full, ndct_noise_full, mask
            )
            
            if pred_patches is None or ndct_patches is None:
                tissue_losses[tissue_name] = 0.0
                continue
            
            # ██ Step 5: Compute NPS for both ██
            nps_pred = self._compute_nps(pred_patches)
            nps_ndct = self._compute_nps(ndct_patches)
            
            # ██ Step 6: Weighted L2 loss between NPS curves ██
            nps_diff = F.mse_loss(nps_pred, nps_ndct)
            
            weighted_diff = weight * nps_diff
            total_loss = total_loss + weighted_diff
            
            tissue_losses[tissue_name] = nps_diff.item()
            n_valid += 1
        
        # Normalize by number of valid tissues
        if n_valid > 0:
            total_loss = total_loss / n_valid
        
        return total_loss, tissue_losses


# ======================================================================
# Convenience: Quick Test
# ======================================================================

if __name__ == "__main__":
    print("Testing AnatomyAwareNPSLoss...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Simulate normalized images [-1, 1]
    B, C, H, W = 2, 1, 512, 512
    ndct = torch.randn(B, C, H, W, device=device) * 0.3  # Low variance (clean)
    predicted = ndct + torch.randn_like(ndct) * 0.1        # Slightly noisy
    predicted.requires_grad_(True)
    
    criterion = AnatomyAwareNPSLoss(
        hu_min=-1000,
        hu_max=1000,
        patch_size=64,
        n_patches=4,
    )
    
    loss, tissue_losses = criterion(predicted, ndct)
    
    print(f"  Total NPS Loss: {loss.item():.6f}")
    for tissue, val in tissue_losses.items():
        print(f"  {tissue:>12}: {val:.6f}")
    
    # Verify gradient flows
    loss.backward()
    assert predicted.grad is not None, "Gradient must flow through predicted!"
    print(f"  ✅ Gradient flows correctly (grad norm: {predicted.grad.norm():.4f})")
    print("  ✅ All tests passed!")
