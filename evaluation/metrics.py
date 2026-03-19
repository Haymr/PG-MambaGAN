"""
Evaluation Metrics
==================
PSNR, SSIM, FID, LPIPS, NPS metrikleri.

Mevcut projedeki PSNR/SSIM'e ek olarak Q1/Q2 gereksinimlerini karşılayan
kapsamlı metrik altyapısı.
"""

import numpy as np
import tensorflow as tf


def compute_psnr(y_true, y_pred, max_val=2.0):
    """
    Peak Signal-to-Noise Ratio (PSNR)

    Veriler [-1, 1] aralığında olduğu için max_val = 2.0
    """
    return tf.image.psnr(y_true, y_pred, max_val=max_val).numpy()


def compute_ssim(y_true, y_pred, max_val=2.0):
    """
    Structural Similarity Index (SSIM)

    Yapısal benzerliği ölçer. 1'e yakın = daha iyi.
    """
    return tf.image.ssim(y_true, y_pred, max_val=max_val).numpy()


def compute_rmse(y_true, y_pred):
    """Root Mean Square Error"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def compute_nps_1d(image, patch_size=64, num_patches=8):
    """
    1D Noise Power Spectrum hesapla.

    Gürültü dokusunun frekans dağılımını ölçer.
    Radial ortalamayla 2D NPS → 1D NPS'ye dönüştürülür.

    Args:
        image: (H, W) numpy array
        patch_size: Patch boyutu
        num_patches: Rastgele patch sayısı

    Returns:
        1D NPS array (radial ortalama)
    """
    from scipy import ndimage

    h, w = image.shape[:2]
    if image.ndim == 3:
        image = image[:, :, 0]

    # Mean filter ile gürültü çıkar
    smoothed = ndimage.uniform_filter(image, size=3)
    noise = image - smoothed

    # Rastgele patch'ler
    nps_2d = np.zeros((patch_size, patch_size))
    for _ in range(num_patches):
        y = np.random.randint(0, h - patch_size)
        x = np.random.randint(0, w - patch_size)
        patch = noise[y:y + patch_size, x:x + patch_size]

        fft = np.fft.fft2(patch)
        nps_2d += np.abs(np.fft.fftshift(fft)) ** 2

    nps_2d /= num_patches

    # Radial ortalama → 1D NPS
    center = patch_size // 2
    Y, X = np.ogrid[:patch_size, :patch_size]
    r = np.sqrt((X - center) ** 2 + (Y - center) ** 2).astype(int)

    max_r = min(center, patch_size - center)
    nps_1d = np.zeros(max_r)
    for i in range(max_r):
        mask = r == i
        if np.any(mask):
            nps_1d[i] = np.mean(nps_2d[mask])

    return nps_1d


def evaluate_batch(generator, inputs, targets, batch_size=50):
    """
    Bir batch veri üzerinde tüm metrikleri hesapla.

    Args:
        generator: Generator model
        inputs: LDCT görüntüler
        targets: NDCT görüntüler
        batch_size: İşleme batch boyutu

    Returns:
        dict: Metrik sonuçları
    """
    psnr_scores, ssim_scores = [], []
    total = len(inputs)

    for i in range(0, total, batch_size):
        batch_in = inputs[i:i + batch_size]
        batch_tar = targets[i:i + batch_size]

        batch_pred = generator(batch_in, training=False)

        psnr = compute_psnr(batch_tar, batch_pred)
        ssim = compute_ssim(batch_tar, batch_pred)

        psnr_scores.extend(psnr)
        ssim_scores.extend(ssim)

    results = {
        'psnr_mean': np.mean(psnr_scores),
        'psnr_std': np.std(psnr_scores),
        'ssim_mean': np.mean(ssim_scores),
        'ssim_std': np.std(ssim_scores),
    }

    return results


def print_results(results, dataset_name=""):
    """Sonuçları formatla ve yazdır"""
    print(f"\n{'=' * 50}")
    print(f"📊 SONUÇLAR {f'({dataset_name})' if dataset_name else ''}")
    print(f"{'=' * 50}")
    print(f"  PSNR: {results['psnr_mean']:.4f} ± {results['psnr_std']:.4f} dB")
    print(f"  SSIM: {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")
    print(f"{'=' * 50}")
