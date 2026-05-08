# 07. File-by-File Reference

This document acts as a dictionary for the codebase. If an LLM needs to know exactly what a file does, this is the reference.

### Root Directory
- **`train.py`**: The main entry point for training. Merges CLI arguments with `configs/default.yaml`, handles hardware setup (VRAM auto-detection via `Environment`), builds the Dataset, Models (G & D), Loss functions, and initializes the `Trainer` class.
- **`evaluate.py`**: The main entry point for post-training Q1-grade clinical evaluation. Runs 2D slice metrics, 3D volumetric assembly, PyRadiomics hallucination checks, and Clinical edge preservation tasks. Outputs full JSON reports.
- **`preprocess.py`**: Converts raw DICOM datasets (Mayo Clinic / PhantomX) into preprocessed `.npy` slices. Clips HU values `[-1000, 1000]`, normalizes to `[-1, 1]`, and exports `[patient_id]_meta.json` files for later NIfTI 3D reconstruction.

### `configs/`
- **`default.yaml`**: The single source of truth for hyperparameters. Defines generator depths, discriminator filters, batch sizes, learning rates, loss lambdas, and dynamic loss schedules.

### `data/`
- **`dataset.py`**: Contains `LDCTDataset`, the PyTorch `Dataset` implementation. Loads paired `.npy` files **assumed to already be normalized to `[-1, 1]` by `preprocess.py`** — the loader only applies a safety clip, no second normalization. Applies runtime data augmentations (flips, rotations).
- **`patient_manifest.py`**: Manages the JSON manifest controlling train/val/test splits cleanly by Patient ID (preventing data leakage across splits).

### `evaluation/`
- **`metrics.py`**: Pure math functions for PSNR, SSIM, RMSE, MAE. Also contains `get_body_mask()` to prevent evaluating background air.
- **`volumetric.py`**: Contains `VolumeAssembler`. Stacks 2D `.npy` slice predictions into a 3D tensor and uses `nibabel` and DICOM affine metadata to export a clinically compliant `.nii.gz` file.
- **`hallucination.py`**: Integrates `pyradiomics` to evaluate if the generator destroyed clinical features.
- **`clinical_task.py`**: Calculates Edge Preservation Index (EPI) and Contrast-to-Noise Ratio (CNR).

### `models/generators/`
- **`vss_unet.py`**: The core Mamba-based generator. Builds the U-Net architecture using VSS stages, PatchMerging, and PatchExpanding. The final convolution before `Tanh` is initialized with `nn.init.xavier_normal_(weight, gain=1.0)` — balanced for `Tanh`'s linear regime, avoiding both saturation (Kaiming) and dead-output collapse (the older `xavier_uniform_(gain=0.1)`).
- **`vss_block.py`**: Contains the low-level State Space Model mathematics (derived from VMamba/mamba-ssm). Defines `VSSStage`, `SSM2d`, and standard Patch operations.
- **`unet_baseline.py`**: A standard CNN-based U-Net kept strictly for ablation studies and benchmarking.

### `models/discriminators/`
- **`patch_disc.py`**: Implements a PatchGAN discriminator. Wraps convolutions in `spectral_norm` to enforce Lipschitz constraints for WGAN-GP.

### `models/losses/`
- **`standard_loss.py`**: Contains `L1Loss`.
- **`anatomy_nps.py`**: The novel Anatomy-Aware NPS Loss. Computes AAPM TG-233 Noise Power Spectrums separately for soft tissue, lung, and fat masks derived entirely from NDCT ground truths. Uses raw `|FFT|²` (no `log1p`), unit-integral shape normalization, and `F.l1_loss` between profiles — comparing spectral *shape*, not magnitude.
- **`frequency_loss.py`**: 2D FFT magnitude loss.
- **`perceptual_loss.py`**: VGG-16 feature-matching loss.

### `training/`
- **`trainer.py`**: The beating heart of the optimization loop.
  - Controls mixed precision (`autocast(bfloat16)`).
  - Handles Gradient Penalty (`autograd.grad` outside of autocast).
  - Manages `.train()` vs `.eval()` mode switching to protect Discriminator SN buffers.
  - Implements the Cosine Annealing loss scheduling.
  - Logs fixed-sample images to W&B using `G_raw`.
  - Maintains `G_ema` weights for checkpointing.

### `setup/`
- **`environment.py`**: Automatically detects if the code is running on Colab (T4), a local workstation (RTX 3080/4090), or a Mac. Adjusts optimal physical batch sizes and gradient accumulation steps to prevent OOM errors dynamically.
