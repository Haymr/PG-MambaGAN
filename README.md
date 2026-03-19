# PG-MambaGAN: Physics-Guided Mamba-WGAN for LDCT Denoising

A novel architecture combining Mamba State Space Models with WGAN-GP adversarial training and physics-guided loss functions for Low-Dose CT image denoising.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🔬 Key Contributions

1. **Mamba-U Generator**: First Mamba-based generator within an adversarial framework for LDCT denoising
2. **Physics-Guided Losses**: NPS preservation + FFT frequency-domain loss integrated with adversarial training
3. **Dose-Adaptive Training**: Single model supporting multiple dose reduction levels

## 📁 Project Structure

```
├── configs/                    # Experiment configurations (YAML)
│   └── default.yaml
├── models/
│   ├── generators/
│   │   ├── unet_baseline.py   # Baseline U-Net (ablation study)
│   │   └── mamba_gen.py        # Mamba-U Generator (proposed)
│   ├── discriminators/
│   │   └── patch_disc.py       # PatchGAN Discriminator
│   └── losses/
│       ├── standard.py         # L1 + Wasserstein + GP
│       ├── perceptual.py       # VGG19 Perceptual Loss
│       └── physics_guided.py   # NPS + FFT Loss (novel)
├── training/
│   └── trainer.py              # PGMambaGAN training loop
├── evaluation/
│   └── metrics.py              # PSNR, SSIM, NPS, FID, LPIPS
├── experiments/                # Experiment results
├── notebooks/                  # Analysis notebooks
├── app/                        # Desktop application
├── weights/                    # Model weights
├── paper/                      # LaTeX manuscript
├── train.py                    # Training entry point
└── requirements.txt
```

## 🚀 Quick Start

### Training

```bash
# PG-MambaGAN (proposed)
python train.py --data-path /path/to/npy_data --config configs/default.yaml

# Baseline U-Net (for comparison)
python train.py --data-path /path/to/npy_data --generator unet_baseline
```

### Evaluation

```python
from evaluation.metrics import evaluate_batch, print_results
results = evaluate_batch(generator, X_test, Y_test)
print_results(results, "Mayo Validation")
```

## 📊 Architecture

```
Input (LDCT) ──► CNN Encoder ──► Mamba Bridge ──► CNN Decoder ──► Output (Denoised)
                    │              (Global)           ▲
                    └──────── Skip Connections ────────┘

Loss = λ₁·L_adv + λ₂·L_L1 + λ₃·L_perceptual + λ₄·L_NPS + λ₅·L_freq
```

## 📝 License

MIT License
