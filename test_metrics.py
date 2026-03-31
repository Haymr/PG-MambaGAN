import numpy as np
from evaluation.metrics import compute_psnr, compute_ssim, extract_body_contour

# Create dummy predicted and target images (e.g. 512x512)
np.random.seed(42)
predicted = np.random.uniform(-1, 1, (512, 512))
target = np.random.uniform(-1, 1, (512, 512))

# Make a simple body contour shape in the target to test extraction
target[100:400, 100:400] = 0.5 # Tissue
target[0:50, :] = -1.0 # Background

# 1. Test contour extraction
mask = extract_body_contour(target)
print("Mask shape:", mask.shape)
print("Mask True count:", np.sum(mask))

# 2. Test compute_psnr with and without mask
psnr_unmasked = compute_psnr(predicted, target)
psnr_masked = compute_psnr(predicted, target, mask=mask)
print("PSNR (unmasked):", psnr_unmasked)
print("PSNR (masked):", psnr_masked)

# 3. Test compute_ssim with and without mask
ssim_unmasked = compute_ssim(predicted, target)
ssim_masked = compute_ssim(predicted, target, mask=mask)
print("SSIM (unmasked):", ssim_unmasked)
print("SSIM (masked):", ssim_masked)

print("Tests passed successfully.")
