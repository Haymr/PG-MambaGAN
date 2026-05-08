# 03. Model Architecture

This document maps out the neural network models inside PG-MambaGAN.

## 1. Why VSS (Visual State Space) Mamba?
Traditional CNNs used in Unet architectures only possess a local receptive field. To capture global context (e.g., the relationship between opposite ribs or the boundary of the lung cavity), CNNs must be extremely deep, causing blurring. Transformers solve this with Self-Attention, but Self-Attention scales quadratically with image resolution $O(N^2)$, which is computationally impossible for medical imaging (e.g., $512 \times 512$ requires $262,144$ tokens). 
State Space Models (Mamba) scale linearly $O(N)$ and act as recurrent systems, theoretically providing an infinite receptive field without the quadratic memory penalty.

## 2. Generator: VSS-U-Net (`models/generators/vss_unet.py`)
The generator replaces traditional convolution blocks with VSS stages.

### Architecture & Tensor Flow (Assuming $512 \times 512$ input)
- **Input:** `(B, 1, 512, 512)`
- **Patch Embedding:** Uses `kernel=3`, `stride=2`, `padding=1` (since patch_size=2). 
  - Output: `(B, 96, 256, 256)`
- **Encoder Stages:**
  - Stage 1: 2 VSS Blocks -> `(B, 96, 256, 256)` -> Skip 1
  - PatchMerging -> `(B, 192, 128, 128)`
  - Stage 2: 2 VSS Blocks -> `(B, 192, 128, 128)` -> Skip 2
  - PatchMerging -> `(B, 384, 64, 64)`
  - Stage 3: 4 VSS Blocks -> `(B, 384, 64, 64)` -> Skip 3
  - PatchMerging -> `(B, 768, 32, 32)`
  - Stage 4 (Bottleneck): 2 VSS Blocks -> `(B, 768, 32, 32)`
- **Decoder Stages:** (Symmetrical upsampling via PatchExpanding, concatenation, and Channel Reduction)
  - PatchExpanding -> `(B, 384, 64, 64)` + Concat Skip 3 -> VSS Blocks
  - PatchExpanding -> `(B, 192, 128, 128)` + Concat Skip 2 -> VSS Blocks
  - PatchExpanding -> `(B, 96, 256, 256)` + Concat Skip 1 -> VSS Blocks
- **Final Upsample (Anti-Checkerboard Head):**
  - Bilinear Upsampling (`scale_factor=2`) -> `(B, 96, 512, 512)`
  - Double Convolution for smoothing:
    - Conv2d(96, 48), GELU
    - Conv2d(48, 48), GELU
    - Conv2d(48, 1), **Tanh**
  - **Output:** `(B, 1, 512, 512)`

### Crucial Weight Initialization
The final convolution layer before `Tanh` is initialized with **`nn.init.xavier_normal_(weight, gain=1.0)`**. Both extremes are dangerous here:
- **Kaiming/ReLU init** (too large) → `Tanh` saturates at `±1.0` → "white dot" artifacts.
- **Xavier Uniform with `gain=0.1`** (too small) → `Tanh` output ≈ 0 → uniform gray/black "dead output", weak gradients, multi-epoch dead zone before learning starts.

`Xavier Normal (gain=1.0)` produces unit-variance pre-activations matched to `Tanh`'s linear regime, allowing the generator to express the full intensity range from the first epoch.

## 3. Discriminator: PatchGAN-SN (`models/discriminators/patch_disc.py`)
To evaluate structural coherence locally rather than globally, PG-MambaGAN uses a PatchGAN architecture.

- **Input:** `(B, 2, 512, 512)` - Channel 0 is the LDCT condition, Channel 1 is the Real (NDCT) or Fake (Generator Output).
- **Layers:** 4 Convolutional layers progressively increasing filters `[64, 128, 256, 512]` with `stride=2`.
- **Output:** `(B, 1, 62, 62)` receptive field map. Each pixel evaluates a local patch of the input image.
- **Spectral Normalization:** Every convolution is wrapped in PyTorch's `spectral_norm`. This strictly bounds the Lipschitz constant of the discriminator, which is a mathematical requirement for stable Wasserstein GAN (WGAN-GP) training. Without SN, the generator will mode-collapse.
- **No Sigmoid:** The final layer is linear because WGAN-GP relies on unbounded Earth Mover's distance, not binary probabilities.
