# 01. Project Overview & Fatal Risks

## What is the project trying to achieve?
The **PG-MambaGAN** project aims to perform ultra-high fidelity Low-Dose Computed Tomography (LDCT) denoising. Traditional Convolutional Neural Networks (CNNs) struggle with capturing long-range anatomical dependencies without suffering from quadratic memory scaling (unlike Transformers). PG-MambaGAN utilizes a **Visual State Space (VSS) Mamba** architecture combined with Generative Adversarial Networks (GANs).

Crucially, this project integrates **Physics-Guided (PG)** constraints. Medical denoising algorithms inherently risk "hallucinating" (inventing) tissues or blurring microcalcifications. PG-MambaGAN explicitly forces the generator to respect the clinical **Noise Power Spectrum (NPS)** and evaluates outputs based on AAPM TG-233 standards. The final objective is to map noisy LDCT scans to clean, Normal-Dose CT (NDCT) quality images without losing diagnostic integrity.

## Fatal Risks in this Domain

When modifying this repository, an LLM/Agent must be acutely aware of the following domain-specific fatal risks. Failing to observe these rules will lead to the silent destruction of the model's clinical validity.

### 1. Checkerboard Artifacts
**Risk:** Generative models using `ConvTranspose2d` for upsampling frequently suffer from checkerboard (grid) artifacts due to uneven overlapping kernels. In medical imaging, this looks like synthetic mesh patterns over tissues.
**Mitigation:** The `vss_unet.py` strictly uses a `Bilinear Upsample -> Dual Convolution (smoothing)` block in the final stage. **Do not reintroduce Transpose Convolutions.**

### 2. Quantum Noise Hallucination (NPS Mask Poisoning)
**Risk:** Computing anatomical tissue masks from the noisy LDCT images or the predicted images causes the quantum noise (random photon scattering) to severely distort Hounsfield Unit (HU) thresholds. This poisons the Anatomy-Aware NPS Loss.
**Mitigation:** Masks must be extracted **EXCLUSIVELY from the NDCT (Ground Truth)** images. These masks dictate *where* the NPS is calculated on the predicted images, but they themselves must never be derived from predicted data.

### 3. Gradient Leakage & SN Buffer Corruption (Mode Collapse)
**Risk:** During the Generator's training step, if the Discriminator (`PatchGAN`) is not explicitly frozen (`requires_grad = False`), the adversarial loss backward pass will write inverse gradients into the Discriminator, destroying its stability.
**Fatal PyTorch Nuance:** Even with `requires_grad = False`, if the Discriminator is left in `.train()` mode during the generator's forward pass, its **Spectral Normalization (SN) Power Iteration vectors** (`weight_u`, `weight_v`) will adapt entirely to the fake data distribution. This ruins the Lipschitz constraint and leads to total mode collapse (the generator outputting solid black or white dots).
**Mitigation:** In `trainer.py`, the discriminator must be explicitly placed in `.eval()` mode during the generator step, and returned to `.train()` mode before its own update step.

### 4. Tanh Saturation vs. Dead Output via Improper Initialization
**Risk:** The final convolutional layer before `Tanh` is sensitive in two opposite directions:
- **Too large weights** (e.g., Kaiming/ReLU init): `Tanh` saturates instantly at `±1.0`, producing "white dot" artifacts early in training.
- **Too small weights** (e.g., `Xavier Uniform` with `gain=0.1`): `Tanh` output collapses to `≈0` (uniform gray/black), gradients vanish, and the generator fails to escape the dead zone for many epochs.

**Mitigation:** The final convolutional layer uses `nn.init.xavier_normal_(weight, gain=1.0)`. This produces a balanced unit-variance initialization that lets `Tanh` operate in its linear regime initially while still allowing the generator to develop structural detail across all intensity ranges.

> **Historical note:** An earlier version of the code used `xavier_uniform_(gain=0.1)`, which was found to crush the initial output to near-zero and required many epochs of L1 pressure before the generator could produce non-trivial outputs. The current `xavier_normal_(gain=1.0)` fixes this dead-output failure mode.
