# PG-MambaGAN

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20596161.svg)](https://doi.org/10.5281/zenodo.20596161)
[![License](https://img.shields.io/badge/License-All_Rights_Reserved-yellow.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12%2B-EE4C2C.svg?logo=pytorch)]()

> 🚨 **Academic Notice: Under Peer Review**
> 
> **IMPORTANT:** This repository contains the official codebase and core implementation of the **PG-MambaGAN** architecture. This manuscript and its comprehensive benchmarks have been submitted and are currently **under peer review** in a top-tier medical imaging journal.
> 
> Independent replication, unauthorized publishing of scientific papers using this exact codebase, or utilizing the proprietary Anatomy-Aware NPS Loss without prior academic consent and proper citation is strictly prohibited during the review process.
> **Physics-Guided Mamba-GAN for Low-Dose CT Denoising**
>
> *The first framework unifying Visual State Space (Mamba) architecture,
> WGAN-GP adversarial stability, and Anatomy-Aware Noise Power Spectrum loss
> for clinically validated CT denoising with hallucination risk control.*

---

## 🔬 Scientific Contribution

PG-MambaGAN introduces three novel contributions to low-dose CT denoising:

| Contribution | Status Quo | Our Approach |
|---|---|---|
| **Architecture** | CNN (U-Net) bottleneck | Full VSS-U-Net — Mamba SSM at every encoder/decoder stage |
| **Loss Function** | Global pixel-wise L1/L2 | **Anatomy-Aware NPS** — tissue-specific spectral matching via NDCT-guided masking |
| **Validation** | PSNR/SSIM only | 3D volumetric continuity + radiomic hallucination testing + clinical task validation |

### Key Design Decisions
- **NDCT-Only Masking**: Tissue segmentation masks derived exclusively from normal-dose ground truth — never from noisy LDCT or model predictions
- **Gradient-Safe Masks**: All mask tensors `detach()`ed from the computational graph — gradients flow through predictions, not mask boundaries
- **Morphological Cleanup**: Binary closing (5×5) → opening (3×3) eliminates noise-induced mask artifacts

---

## 🏗️ Architecture

### VSS-U-Net Generator

```
LDCT (1, 512, 512)
  │
  ├─ Patch Embed ─── Conv 3×3, stride 2, pad 1 ──→ (96, 256, 256)
  │
  ├─ Encoder ──────────────────────────────────────────────
  │  Stage 1:  2× VSS Block, dim=96   (256×256) ──→ skip₁
  │  ↓ PatchMerging
  │  Stage 2:  2× VSS Block, dim=192  (128×128) ──→ skip₂
  │  ↓ PatchMerging
  │  Stage 3:  4× VSS Block, dim=384  (64×64)   ──→ skip₃
  │  ↓ PatchMerging
  │  Stage 4:  2× VSS Block, dim=768  (32×32)   Bottleneck
  │
  ├─ Decoder ──────────────────────────────────────────────
  │  ↑ PatchExpanding + skip₃ → 4× VSS, dim=384  (64×64)
  │  ↑ PatchExpanding + skip₂ → 2× VSS, dim=192  (128×128)
  │  ↑ PatchExpanding + skip₁ → 2× VSS, dim=96   (256×256)
  │
  └─ Head ─── Bilinear Upsample → Conv 3×3 → Conv 3×3 (Xavier Normal) → Tanh
               ──→ Denoised (1, 512, 512)
```

Each **VSS Block** performs 4-way 2D Selective Scan (SS2D) via `mamba-ssm`:
- Uses **Spatially Coherent Z-shaped Scanning** (Snake Scan) instead of default raster scans to preserve spatial continuity of non-linear medical structures.
- Z-scan (→↓), Reverse Z-scan (←↑), Column Z-scan (↓→), Reverse Column Z-scan (↑←)
- O(n) complexity vs O(n²) for self-attention

### Empirical Deprecations (Ablation Insights)
During clinical-grade 512×512 scaling, we formally deprecated the following standard GAN conventions based on empirical WandB data:
- **Spectral Normalization Disabled**: SN was removed from the Discriminator. When combined with WGAN-GP and aggressive NPS FFT losses, SN artificially choked the Discriminator's capacity, causing gradients to explode (e.g., GP loss spiking to 3800+) and Discriminator loss to permanently flatline at ~0.01.
- **Xavier Normal Initialization**: Generator's final `Tanh` convolution is explicitly initialized with `Xavier Normal (gain=1.0)`. `Kaiming` caused severe Tanh saturation (white dot artifacts), while `Xavier Uniform (gain=0.1)` suppressed variance entirely, leading to dead gradients and uniform gray outputs for multiple epochs.
- **NPS Shape Normalization**: `log1p` compression was dropped from the NPS loss as it crushed dynamic range. Both predicted and ground-truth spectra are now normalized to **unit-integral** before comparison, targeting the physical *shape* of the noise distribution rather than absolute magnitude.

### Anatomy-Aware NPS Loss Pipeline

```
NDCT (Ground Truth)                    Predicted (Generator Output)
        │                                       │
        ▼                                       │
  ┌─ HU Denormalize ─┐                         │
  │  [-1,1] → HU     │                         │
  └───────┬───────────┘                         │
          ▼                                     │
  ┌─ HU Thresholding ─────────────────┐         │
  │  Air < -900 | Lung -900~-500      │         │
  │  Fat -500~-100 | Soft -100~300    │         │
  │  Bone ≥ 300                       │         │
  └───────┬───────────────────────────┘         │
          ▼                                     │
  ┌─ Morphological Cleanup ───────────┐         │
  │  Closing (5×5) → Opening (3×3)    │         │
  │  Min area filter (64 px)          │         │
  └───────┬───────────────────────────┘         │
          ▼                                     │
  ┌─ AAPM TG-233 Homogeneity Filter ──┐         │
  │  3x3 Sobel Gradient Magnitude     │         │
  │  Fat (<0.02) | Soft (<0.04)       │         │
  │  Lung (<0.08)                     │         │
  │  *Reject patches with anatomy*    │         │
  └───────┬───────────────────────────┘         │
          ▼                                     │
   Tissue Masks (.detach())  ───────────┬───────┘
          │                             │
          ▼                             ▼
    NDCT ⊙ Mask                  Pred ⊙ Mask
          │                             │
          ▼                             ▼
    NPS(NDCT)                    NPS(Pred)     ← 1st-Order Detrending → |FFT|² → Radial Avg → Unit-Integral
          │                             │
          └──────── L₁ Loss ────────────┘
                      ×
              Tissue Weight (w_soft=2.0, w_lung=1.5, w_bone=0.0)
```

---

## 📁 Project Structure

```
PG-MambaGAN/
├── configs/
│   └── default.yaml              # VRAM-optimized training config
├── data/
│   ├── dataset.py                # PyTorch Dataset + augmentation
│   └── patient_manifest.py       # Patient-level split (zero leakage)
├── models/
│   ├── generators/
│   │   ├── vss_block.py          # SS2D + Mamba SSM kernel
│   │   ├── vss_unet.py           # Full VSS-U-Net generator
│   │   └── unet_baseline.py   # CNN baseline (ablation)
│   ├── discriminators/
│   │   └── patch_disc.py      # PatchGAN (WGAN-GP, SN disabled)
│   └── losses/
│       ├── anatomy_nps.py        # ★ Anatomy-Aware NPS Loss
│       ├── frequency_loss.py       # Multi-scale FFT loss
│       ├── perceptual_loss.py      # VGG19 + LPIPS
│       └── standard_loss.py        # L1, Wasserstein, GP
├── training/
│   └── trainer.py             # VRAM-optimized WGAN-GP trainer
├── evaluation/
│   ├── metrics.py                # 2D + 3D metrics
│   ├── volumetric.py             # 3D NIfTI assembly
│   ├── hallucination.py          # Radiomic preservation test
│   └── clinical_task.py          # EPI + CNR validation
├── setup/
│   ├── environment.py            # Dual-env detection + VRAM profiling
│   └── colab_setup.py            # One-click Colab setup

├── notebooks/
│   └── train_colab.ipynb         # Google Colab training notebook
├── preprocess.py                 # DICOM → NPY (512×512) + metadata
├── train.py                      # Training entry point
├── evaluate.py                   # Full evaluation pipeline
└── requirements.txt              # PyTorch ecosystem
```

---

## ⚡ Quick Start

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (required for `mamba-ssm`)
- GPU with ≥16GB VRAM (T4 minimum, A100 recommended)

### Installation

```bash
# Clone
git clone https://github.com/Haymr/PG-MambaGAN.git
cd PG-MambaGAN

# Install dependencies
pip install -r requirements.txt

# Install Mamba SSM (requires CUDA)
pip install causal-conv1d>=1.2.0
pip install mamba-ssm>=1.2.0
```

> **⚠️ Mamba CUDA Compilation**: `mamba-ssm` requires a CUDA toolkit matching your
> PyTorch CUDA version. On Colab, run `python setup/colab_setup.py` for automatic setup.

### Google Colab

```python
# In a Colab notebook cell:
!git clone https://github.com/Haymr/PG-MambaGAN.git
%cd PG-MambaGAN
!python setup/colab_setup.py
```

Or use the pre-built notebook: `notebooks/train_colab.ipynb`

---

## 🔧 Usage

### 1. Preprocess DICOM Data

```bash
# Mayo Clinic dataset (512×512, patient-level manifest)
python preprocess.py \
    --input-dir /path/to/LDCT-and-Projection-data \
    --output-dir /path/to/processed \
    --img-size 512 \
    --create-manifest

# PhantomX external dataset
python preprocess.py \
    --input-dir /path/to/phantomx \
    --output-dir /path/to/phantomx_processed \
    --dataset-type phantomx \
    --create-manifest
```

### 2. Train

```bash
# Full VSS-U-Net training (VRAM-optimized)
python train.py \
    --config configs/default.yaml \
    --data-path /path/to/processed

# Resume from checkpoint
python train.py \
    --config configs/default.yaml \
    --data-path /path/to/processed \
    --resume experiments/checkpoints/latest.pth

# Override VRAM settings (e.g., for A100)
python train.py \
    --config configs/default.yaml \
    --data-path /path/to/processed \
    --batch-size 4 \
    --accumulation 2
```

### 3. Evaluate

```bash
# Full Q1-grade evaluation pipeline
python evaluate.py \
    --checkpoint experiments/checkpoints/best.pth \
    --data-path /path/to/processed \
    --output-dir experiments/evaluation

# Skip specific evaluations
python evaluate.py \
    --checkpoint experiments/checkpoints/best.pth \
    --data-path /path/to/processed \
    --skip-hallucination \
    --skip-clinical
```

### 4. Ablation Study

```bash
# CNN Baseline (standard U-Net for comparison)
# Edit configs/default.yaml: generator: "unet_baseline"
python train.py --config configs/default.yaml --data-path /path/to/processed

# Without NPS Loss (set nps_start: 0.0, nps_end: 0.0 in config)
# Without Perceptual Loss (set lambda_perceptual: 0.0 in config)
```

---

## 📊 VRAM Optimization

PG-MambaGAN implements three VRAM management strategies to enable
512×512 VSS-U-Net training on consumer GPUs (16GB):

| Strategy | VRAM Savings | Config Key |
|---|---|---|
| **BFloat16 AMP** | ~40% | `training.mixed_precision: true` |
| **Gradient Checkpointing** | ~60% | `generator.gradient_checkpointing: true` |
| **Gradient Accumulation** | Batch-independent | `training.gradient_accumulation: 8` |

> **Critical**: WGAN-GP's Gradient Penalty is computed **outside** the AMP autocast
> scope in pure FP32 to prevent NaN from second-order gradient computation.
> 
> **Stability Core Fixes**: The PyTorch trainer overrides native `requires_grad` rules to prevent silent `gradient_checkpointing` graph disconnections (freezing), and explicitly synchronizes `G_ema` module buffers to prevent running metric validation drift across epochs.

### Recommended VRAM Configurations

| GPU | VRAM | batch_size | accumulation | Effective Batch | Notes |
|---|---|---|---|---|---|
| T4 | 16 GB | 1 | 8 | 8 | Minimum fallback configuration |
| RTX 3080/4090 | 10-24 GB | 8 | 1 | 8 | Standard. AMP keeps 512×512 peak VRAM ~12.6GB |
| A100 | 40 GB | 16 | 1 | 16 | High-throughput batching |

> **Note on EMA Decay**: Exponential Moving Average (EMA) decay is tuned to `0.99` rather than the conventional `0.999`. A decay of `0.999` created too much lag for diagnostic logging, rendering visual feedback on W&B "stale" relative to the actively learning Generator.

---

## 🔬 Evaluation Pipeline

The evaluation pipeline produces four categories of evidence:

1. **2D Per-Slice**: PSNR, SSIM, RMSE, MAE (computed with body contouring masks to prevent background air inflation)
   *Note: MAE (L1) and Best Checkpoint saving dynamically use a dual-logging system (Normalized Space vs Physical Hounsfield Space) for direct clinical interpretability.*
2. **3D Volumetric**: Multi-planar (Axial, Coronal, Sagittal) 3D-SSIM/3D-PSNR, plus **Flickering Index** (z-axis continuity)
3. **Hallucination Risk**: Radiomic feature preservation (First Order, forced 2D-GLCM, GLRLM)
4. **Clinical Validity**: Edge Preservation Index (EPI), Contrast-to-Noise Ratio (CNR)

All 3D volumes are exported as NIfTI (`.nii.gz`) with DICOM-derived affine matrices
preserving `PixelSpacing`, `SliceThickness`, and `ImagePositionPatient` metadata.
Slices are sorted by **Z-coordinate** (not filename index) to handle Head-First/Feet-First orientation differences.

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `torch` | ≥2.1.0 | Core framework |
| `mamba-ssm` | ≥1.2.0 | Selective State Space kernel (CUDA required) |
| `causal-conv1d` | ≥1.2.0 | Causal convolution for Mamba |
| `monai` | ≥1.3.0 | Medical image transforms |
| `nibabel` | ≥5.0.0 | NIfTI I/O with affine preservation |
| `pyradiomics` | ≥3.1.0 | Radiomic feature extraction |
| `wandb` | ≥0.15.0 | Experiment tracking |
| `scikit-image` | ≥0.21.0 | SSIM, PSNR metrics |
| `pydicom` | ≥2.4.0 | DICOM preprocessing |
| `opencv-python` | ≥4.8.0 | Image resizing |
| `lpips` | ≥0.1.4 | Perceptual similarity (optional) |

---

## 📄 Total Loss Function

```text
L_total = λ_adv  · L_wasserstein      (1.0)                        — WGAN-GP adversarial
        + λ_l1   · L_l1               (Cosine Decay: 50.0 → 15.0)  — Pixel fidelity
        + λ_perc · L_perceptual       (10.0)                       — VGG19 feature matching
        + λ_nps  · L_anatomy_nps      (Cosine Warmup: 2.0 → 15.0)  — ★ Tissue-specific NPS
        + λ_freq · L_frequency        (1.0)                        — Multi-scale FFT
```

> **Dynamic Loss Weighting:** To prevent simple L1 pixel loss from dominating the early training phase, the L1 weight dynamically decreases while the Anatomy-Aware NPS weight increases over time. This enables early-stage structural alignment followed by late-stage fine-grained tissue texture learning.

---

## 📖 Citation

```bibtex
@article{pg-mambagan-2026,
  title={PG-MambaGAN: Physics-Guided Visual State Space GAN for 
         Anatomy-Aware Low-Dose CT Denoising},
  author={Çayırcı, Serhan Ege and Akkurt, Şinasi Onuralp and Aki, Koray},
  journal={Under Peer Review},
  year={2026}
}
```

---

## 🙏 Acknowledgments

- Mayo Clinic for the [LDCT-and-Projection-data](https://wiki.cancerimagingarchive.net/display/Public/LDCT-and-Projection-data) dataset
- [VMamba](https://github.com/MzeroMiko/VMamba) for Visual State Space inspiration
- [mamba-ssm](https://github.com/state-spaces/mamba) for the selective scan CUDA kernel

---

## 📜 License

**All Rights Reserved.**

This software is currently withheld from open licensing pending academic peer-review publication. Unauthorized academic publishing or replication is strictly prohibited. Full open-source licensing (MIT) will be restored upon manuscript acceptance. See [LICENSE](LICENSE) for details.
