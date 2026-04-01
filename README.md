# PG-MambaGAN

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
  ├─ Patch Embed ─── Conv 4×4, stride 4 ──→ (96, 128, 128)
  │
  ├─ Encoder ──────────────────────────────────────────────
  │  Stage 1:  2× VSS Block, dim=96   (128×128) ──→ skip₁
  │  ↓ PatchMerging
  │  Stage 2:  2× VSS Block, dim=192  (64×64)   ──→ skip₂
  │  ↓ PatchMerging
  │  Stage 3:  4× VSS Block, dim=384  (32×32)   ──→ skip₃
  │  ↓ PatchMerging
  │  Stage 4:  2× VSS Block, dim=768  (16×16)   Bottleneck
  │
  ├─ Decoder ──────────────────────────────────────────────
  │  ↑ PatchExpanding + skip₃ → 4× VSS, dim=384  (32×32)
  │  ↑ PatchExpanding + skip₂ → 2× VSS, dim=192  (64×64)
  │  ↑ PatchExpanding + skip₁ → 2× VSS, dim=96   (128×128)
  │
  └─ Head ─── TransposeConv 4×4 → Conv 1×1 → Tanh
               ──→ Denoised (1, 512, 512)
```

Each **VSS Block** performs 4-way 2D Selective Scan (SS2D) via `mamba-ssm`:
- Raster scan (→↓), Reverse raster (←↑), Column scan (↓→), Reverse column (↑←)
- O(n) complexity vs O(n²) for self-attention

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
  │  Min area filter (64² px)         │         │
  └───────┬───────────────────────────┘         │
          ▼                                     │
   Tissue Masks (.detach())  ───────────┬───────┘
          │                             │
          ▼                             ▼
    NDCT ⊙ Mask                  Pred ⊙ Mask
          │                             │
          ▼                             ▼
    NPS(NDCT)                    NPS(Pred)     ← 2D FFT → Radial Avg
          │                             │
          └──────── L₂ Loss ────────────┘
                      ×
              Tissue Weight (w_soft=2.0, w_lung=1.5, ...)
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
│   │   └── unet_baseline_pt.py   # CNN baseline (ablation)
│   ├── discriminators/
│   │   └── patch_disc_pt.py      # SN-PatchGAN (WGAN-GP)
│   └── losses/
│       ├── anatomy_nps.py        # ★ Anatomy-Aware NPS Loss
│       ├── frequency_pt.py       # Multi-scale FFT loss
│       ├── perceptual_pt.py      # VGG19 + LPIPS
│       └── standard_pt.py        # L1, Wasserstein, GP
├── training/
│   └── trainer_pt.py             # VRAM-optimized WGAN-GP trainer
├── evaluation/
│   ├── metrics.py                # 2D + 3D metrics
│   ├── volumetric.py             # 3D NIfTI assembly
│   ├── hallucination.py          # Radiomic preservation test
│   └── clinical_task.py          # EPI + CNR validation
├── setup/
│   ├── environment.py            # Dual-env detection + VRAM profiling
│   └── colab_setup.py            # One-click Colab setup
├── legacy/                       # Original TF code (archived)
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

# Without NPS Loss (set lambda_nps: 0.0 in config)
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

### Recommended VRAM Configurations

| GPU | VRAM | batch_size | accumulation | Effective Batch |
|---|---|---|---|---|
| T4 | 16 GB | 1 | 8 | 8 |
| RTX 3080 | 10 GB | 1 | 8 | 8 |
| A100 | 40 GB | 4 | 2 | 8 |
| A100 | 80 GB | 8 | 1 | 8 |

---

## 🔬 Evaluation Pipeline

The evaluation pipeline produces four categories of evidence:

1. **2D Per-Slice**: PSNR, SSIM, RMSE, MAE (computed with body contouring masks to prevent background air inflation)
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

```
L_total = λ_adv  · L_wasserstein      (1.0)    — WGAN-GP adversarial
        + λ_l1   · L_l1               (100.0)  — Pixel fidelity
        + λ_perc · L_perceptual        (10.0)  — VGG19 feature matching
        + λ_nps  · L_anatomy_nps       (5.0)   — ★ Tissue-specific NPS
        + λ_freq · L_frequency          (1.0)  — Multi-scale FFT
```

---

## 📖 Citation

```bibtex
@article{pg-mambagan-2026,
  title={PG-MambaGAN: Physics-Guided Visual State Space GAN for 
         Anatomy-Aware Low-Dose CT Denoising},
  author={[Authors]},
  journal={[Target: IEEE TMI / Medical Image Analysis]},
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

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
