# 04. Loss Functions & Methodology

This document outlines the ensemble of loss functions used to train the generator in PG-MambaGAN. 

## Total Generator Loss Equation
`G_Loss = (lambda_l1 * L1) + (lambda_adv * Adversarial) + (lambda_perceptual * Perceptual) + (lambda_freq * Frequency) + (lambda_nps * NPS)`

### 1. L1 Loss (`models/losses/standard_loss.py`)
- **Purpose:** Forces absolute structural fidelity between the generated image and the NDCT. L1 avoids the blurring effect common with MSE (L2) loss.
- **Dynamic Scheduling:** In `configs/default.yaml`, L1 uses a cosine annealing schedule. `l1_start = 50.0`, `l1_end = 15.0`.
- **Reasoning:** The network is heavily constrained by L1 early in training so it learns basic anatomy first without hallucinating textures. Once the anatomy is stable, L1 relaxes to allow the GAN and NPS losses to develop the texture.

### 2. Adversarial Loss (WGAN-GP)
- **Purpose:** Replaces standard BCE loss with Wasserstein distance. The discriminator acts as a critic attempting to maximize the distance between Real and Fake distributions.
- **Gradient Penalty (`lambda=10.0`):** Enforces the 1-Lipschitz continuity required by WGAN. Computed via `torch.autograd.grad(..., create_graph=True)`.
- **Safety Rule:** The GP calculation must occur outside of `torch.amp.autocast()` and strictly in FP32. Calculating 2nd-order derivatives in BFloat16 causes `NaN` explosions.

### 3. Perceptual Loss (`models/losses/perceptual_loss.py`)
- **Purpose:** Extracted from a pre-trained VGG16 network. It calculates the L1 difference of feature maps. This ensures the denoised image "looks" semantically similar to the NDCT.
- **Weight:** `lambda_perceptual = 10.0`

### 4. Frequency Loss (`models/losses/frequency_loss.py`)
- **Purpose:** Forces the generator to match the 2D FFT magnitude spectrum of the target. Prevents the generator from missing high-frequency edge information.
- **Weight:** `lambda_freq = 1.0`

### 5. Anatomy-Aware NPS Loss (`models/losses/anatomy_nps.py`) (Novelty)
- **Purpose:** Evaluates the physical Noise Power Spectrum (NPS) radially based on AAPM TG-233 standards. It ensures the generative noise distribution matches true quantum mottle.
- **Dynamic Scheduling:** `nps_start = 2.0`, `nps_end = 15.0`.
- **Reasoning:** NPS loss is highly unstable if introduced early before structural convergence. It ramps up as L1 ramps down ("Anatomy first, texture later").
- **Tissue Weights:** Computed separately per tissue using masks derived strictly from NDCT.
  - `soft_tissue: 2.0` (Liver/Tumors - Most important)
  - `lung: 1.5`
  - `fat: 1.0`
  - `bone: 0.0` (Ignored due to high structural density)
  - `air: 0.0` (Ignored)
- **Detrending:** Extracted patches undergo 1st-order (plane) detrending via pseudoinverse to remove structural background trends and isolate pure quantum noise.
- **Power Spectrum:** Raw `|FFT|²` is used directly (no `log1p` compression). Earlier versions applied `torch.log1p(power)` which crushed the spectrum dynamic range and made the gradient signal too small for the generator to react to.
- **Shape Normalization:** Both `nps_pred` and `nps_ndct` are normalized to **unit integral** (`nps / nps.sum()`) before comparison. This converts the loss from a magnitude-matching problem (which is dominated by absolute noise level) to a **shape-matching problem** that targets the spectral *profile* — the actual physical signature of CT noise per AAPM TG-233.
- **Distance Metric:** `F.l1_loss` between unit-integral profiles. L1 is more robust to outlier frequency bins than MSE/L2 and is consistent with the L1-style "shape distance" used in NPS literature.
