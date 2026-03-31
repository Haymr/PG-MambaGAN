"""
PG-MambaGAN — Evaluation Metrics (2D + 3D)

Standard image quality metrics and 3D volumetric metrics.

2D Per-Slice:
    - PSNR, SSIM, RMSE, MAE

3D Volumetric (Revision #3):
    - 3D-SSIM (slice-by-slice averaged)
    - Flickering Index (z-axis continuity)
    - VIF (Visual Information Fidelity)
"""

import numpy as np
from typing import Dict, Optional, Tuple
import scipy.ndimage

try:
    from skimage.metrics import (
        peak_signal_noise_ratio as _psnr,
        structural_similarity as _ssim,
    )
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


# ======================================================================
# 2D Per-Slice Metrics
# ======================================================================

def get_body_mask(target_hu: np.ndarray) -> np.ndarray:
    """
    Generate a binary mask of the patient's body region using a -500 HU threshold
    combined with morphological closing and filling.
    """
    mask = target_hu > -500
    mask = scipy.ndimage.binary_closing(mask, structure=np.ones((5, 5)))
    mask = scipy.ndimage.binary_fill_holes(mask)
    return mask


def compute_psnr(
    predicted: np.ndarray,
    target: np.ndarray,
    data_range: float = 2.0,
    body_mask: Optional[np.ndarray] = None,
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.
    
    Args:
        predicted: Predicted image (H, W) in [-1, 1].
        target: Ground truth image (H, W) in [-1, 1].
        data_range: Dynamic range (2.0 for [-1, 1]).
        body_mask: Optional boolean mask of the body region.
    """
    if body_mask is not None:
        mse = np.mean((predicted[body_mask == 1] - target[body_mask == 1]) ** 2)
        if mse == 0:
            return float("inf")
        return float(10 * np.log10(data_range ** 2 / mse))

    if SKIMAGE_AVAILABLE:
        return float(_psnr(target, predicted, data_range=data_range))
    
    mse = np.mean((predicted - target) ** 2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(data_range ** 2 / mse))


def compute_ssim(
    predicted: np.ndarray,
    target: np.ndarray,
    data_range: float = 2.0,
    win_size: int = 11,
    body_mask: Optional[np.ndarray] = None,
) -> float:
    """
    Compute Structural Similarity Index.
    
    Args:
        predicted: Predicted image (H, W) in [-1, 1].
        target: Ground truth image (H, W) in [-1, 1].
        data_range: Dynamic range (2.0 for [-1, 1]).
        win_size: SSIM window size (must be odd, ≤ image dim).
        body_mask: Optional boolean mask of the body region.
    """
    # 2D spatial SSIM using an 11x11 sliding Gaussian window (sigma=1.5)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    
    mu_x = scipy.ndimage.gaussian_filter(predicted, sigma=1.5)
    mu_y = scipy.ndimage.gaussian_filter(target, sigma=1.5)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x2 = scipy.ndimage.gaussian_filter(predicted ** 2, sigma=1.5) - mu_x_sq
    sigma_y2 = scipy.ndimage.gaussian_filter(target ** 2, sigma=1.5) - mu_y_sq
    sigma_xy = scipy.ndimage.gaussian_filter(predicted * target, sigma=1.5) - mu_xy
    
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / \
               ((mu_x_sq + mu_y_sq + c1) * (sigma_x2 + sigma_y2 + c2))

    if body_mask is not None:
        return float(np.mean(ssim_map[body_mask == 1]))
    return float(np.mean(ssim_map))


def compute_rmse(predicted: np.ndarray, target: np.ndarray, body_mask: Optional[np.ndarray] = None) -> float:
    """Root Mean Squared Error."""
    if body_mask is not None:
        return float(np.sqrt(np.mean((predicted[body_mask == 1] - target[body_mask == 1]) ** 2)))
    return float(np.sqrt(np.mean((predicted - target) ** 2)))


def compute_mae(predicted: np.ndarray, target: np.ndarray, body_mask: Optional[np.ndarray] = None) -> float:
    """Mean Absolute Error."""
    if body_mask is not None:
        return float(np.mean(np.abs(predicted[body_mask == 1] - target[body_mask == 1])))
    return float(np.mean(np.abs(predicted - target)))


def compute_2d_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    data_range: float = 2.0,
    body_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute all 2D per-slice metrics.
    
    Args:
        predicted: Predicted image (H, W).
        target: Ground truth image (H, W).
        data_range: Dynamic range.
        body_mask: Optional boolean mask of the body region.
    
    Returns:
        Dict with PSNR, SSIM, RMSE, MAE.
    """
    return {
        "psnr": compute_psnr(predicted, target, data_range, body_mask),
        "ssim": compute_ssim(predicted, target, data_range, win_size=7, body_mask=body_mask),
        "rmse": compute_rmse(predicted, target, body_mask),
        "mae": compute_mae(predicted, target, body_mask),
    }


# ======================================================================
# 3D Volumetric Metrics (Revision #3)
# ======================================================================

def compute_3d_ssim(
    pred_volume: np.ndarray,
    target_volume: np.ndarray,
    data_range: float = 2.0,
) -> Dict[str, float]:
    """
    Compute 3D SSIM: slice-by-slice SSIM averaged over the volume.
    
    Args:
        pred_volume: Predicted volume (N, H, W).
        target_volume: Ground truth volume (N, H, W).
    
    Returns:
        Dict with mean_ssim, std_ssim, min_ssim.
    """
    n_slices = pred_volume.shape[0]
    ssim_values = []
    
    for i in range(n_slices):
        s = compute_ssim(pred_volume[i], target_volume[i], data_range)
        ssim_values.append(s)
    
    ssim_arr = np.array(ssim_values)
    
    return {
        "ssim_3d_mean": float(ssim_arr.mean()),
        "ssim_3d_std": float(ssim_arr.std()),
        "ssim_3d_min": float(ssim_arr.min()),
        "ssim_3d_max": float(ssim_arr.max()),
        "ssim_per_slice": ssim_values,
    }


def compute_flickering_index(
    pred_volume: np.ndarray,
    target_volume: np.ndarray,
) -> Dict[str, float]:
    """
    Compute Flickering Index for z-axis continuity.
    
    Measures the difference between adjacent slices in the predicted volume
    vs the target volume. A good denoiser should NOT introduce inter-slice
    intensity jumps that don't exist in the target.
    
    Flickering Index = mean(|∆pred[z]| - |∆target[z]|)
    where ∆[z] = slice[z+1] - slice[z]
    
    Lower FI = better z-axis continuity.
    Negative FI = over-smoothing along z (also problematic).
    
    Args:
        pred_volume: Predicted volume (N, H, W).
        target_volume: Target volume (N, H, W).
    
    Returns:
        Dict with flickering metrics.
    """
    n_slices = pred_volume.shape[0]
    
    if n_slices < 2:
        return {"flickering_index": 0.0, "flickering_std": 0.0}
    
    # Compute inter-slice differences
    pred_diffs = []
    target_diffs = []
    fi_per_pair = []
    
    for z in range(n_slices - 1):
        temporal_diff_pred = pred_volume[z + 1] - pred_volume[z]
        temporal_diff_target = target_volume[z + 1] - target_volume[z]

        pred_diffs.append(np.mean(np.abs(temporal_diff_pred)))
        target_diffs.append(np.mean(np.abs(temporal_diff_target)))
        fi_per_pair.append(np.mean(np.abs(temporal_diff_pred - temporal_diff_target)))
    
    pred_diffs = np.array(pred_diffs)
    target_diffs = np.array(target_diffs)
    fi_per_pair = np.array(fi_per_pair)
    
    return {
        "flickering_index": float(np.mean(fi_per_pair)),
        "flickering_std": float(np.std(fi_per_pair)),
        "flickering_max": float(np.max(np.abs(fi_per_pair))),
        "pred_z_smoothness": float(np.mean(pred_diffs)),
        "target_z_smoothness": float(np.mean(target_diffs)),
    }


def compute_3d_psnr(
    pred_volume: np.ndarray,
    target_volume: np.ndarray,
    data_range: float = 2.0,
) -> Dict[str, float]:
    """Compute 3D PSNR: slice-by-slice averaged."""
    n_slices = pred_volume.shape[0]
    psnr_values = []
    
    for i in range(n_slices):
        p = compute_psnr(pred_volume[i], target_volume[i], data_range)
        psnr_values.append(p)
    
    psnr_arr = np.array(psnr_values)
    
    return {
        "psnr_3d_mean": float(psnr_arr.mean()),
        "psnr_3d_std": float(psnr_arr.std()),
        "psnr_3d_min": float(psnr_arr.min()),
        "psnr_per_slice": psnr_values,
    }


def compute_volumetric_metrics(
    pred_volume: np.ndarray,
    target_volume: np.ndarray,
    data_range: float = 2.0,
) -> Dict[str, float]:
    """
    Compute all volumetric (3D) metrics.
    
    Args:
        pred_volume: Predicted volume (N, H, W).
        target_volume: Target volume (N, H, W).
    
    Returns:
        Combined dict of 3D-PSNR, 3D-SSIM, and Flickering Index.
    """
    results = {}
    
    # 3D PSNR
    psnr_3d = compute_3d_psnr(pred_volume, target_volume, data_range)
    results.update({k: v for k, v in psnr_3d.items() if "per_slice" not in k})
    
    # 3D SSIM
    ssim_3d = compute_3d_ssim(pred_volume, target_volume, data_range)
    results.update({k: v for k, v in ssim_3d.items() if "per_slice" not in k})
    
    # Flickering Index
    fi = compute_flickering_index(pred_volume, target_volume)
    results.update(fi)
    
    return results
