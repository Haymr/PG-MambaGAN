# 02. Data Handling & Preprocessing

This document details the handling of raw Computed Tomography (CT) data, from DICOM files to `.npy` matrices ready for PyTorch ingestion. Medical tensors are not like standard JPEGs/PNGs; they possess physical meaning (Hounsfield Units). 

## 1. Hounsfield Unit (HU) Conversion
DICOM pixel arrays inherently store raw attenuation coefficients. They must be mathematically mapped to the standard Hounsfield Unit (HU) scale representing radiodensity.
The conversion in `preprocess.py` relies on DICOM tags:
```python
hu = pixel_array * RescaleSlope + RescaleIntercept
```

## 2. Clinical Windowing and Normalization
To feed these physical numbers to a neural network, they are clipped to a clinically relevant window and then linearly normalized to the `[-1, 1]` tensor space suitable for `Tanh` activations.

- **Clipping Range:** `hu_min = -1000` (Air) to `hu_max = 1000` (Dense Bone). Values outside this window are clamped.
- **Normalization Formula:**
  1. Scale to `[0, 1]`: `(hu - hu_min) / (hu_max - hu_min)`
  2. Scale to `[-1, 1]`: `value * 2 - 1`

> **Where this happens:** The HU-to-`[-1, 1]` normalization is performed **once, in `preprocess.py`**, when DICOM files are converted to `.npy` and written to `processed_*/`. The runtime loader (`data/dataset.py`) assumes the `.npy` files are already in `[-1, 1]` and only applies a safety `np.clip(x, -1.0, 1.0)`. **Do not re-add HU-style normalization to `dataset.py`** — an earlier version did this and silently double-normalized the data, corrupting the input distribution and stalling training.

**Denormalization:**
During evaluation or physical loss calculation (like the NPS loss), the network output `[-1, 1]` is mathematically reversed back to standard HU:
```python
hu = (norm + 1.0) / 2.0 * (hu_max - hu_min) + hu_min
```

## 3. Metadata Preservation for 3D Assemblies
To satisfy clinical deployment and radiological integrity, PG-MambaGAN does not discard spatial mapping parameters. When processing slices in `preprocess.py`, we explicitly export a `[patient_id]_meta.json` file.
The JSON includes:
- `PixelSpacing` (In-plane X/Y resolution in mm)
- `SliceThickness` & `SpacingBetweenSlices` (Z-axis resolution in mm)
- `ImagePositionPatient` & `ImageOrientationPatient` (Affine geometry)
This allows the `VolumeAssembler` to recreate clinically accurate `NIfTI` (`.nii.gz`) files from the generated 2D slices.

## 4. Morphological Cleaning for Masks
When extracting anatomical segmentations (e.g., separating Soft Tissue from Lung based on HU brackets in `models/losses/anatomy_nps.py`), isolated pixels and small holes occur.
Instead of relying on heavy semantic segmentation models, PG-MambaGAN uses deterministic, differentiable Morphological Operations via `F.max_pool2d`:
1. **Dilation:** `F.max_pool2d(mask, kernel)` expands the mask.
2. **Erosion:** `-F.max_pool2d(-mask, kernel)` shrinks the mask.
3. **Closing (Dilation -> Erosion):** Fills small holes within a tissue class (Kernel size: 5).
4. **Opening (Erosion -> Dilation):** Removes isolated noise pixels (Kernel size: 3).

*Note: Morphological operations are applied to detached masks to prevent disrupting the computational graph of the generator.*
