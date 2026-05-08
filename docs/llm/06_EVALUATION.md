# 06. Evaluation Metrics

PG-MambaGAN uses a rigid, 5-stage evaluation pipeline to ensure clinical viability, defined in `evaluate.py`. 

## 1. 2D Per-Slice Metrics
Computed on isolated slices (`B, 1, 512, 512`).
- **PSNR (Peak Signal-to-Noise Ratio):** Measures absolute pixel fidelity.
- **SSIM (Structural Similarity Index):** Evaluates perceived visual structure.
- **RMSE / MAE:** Standard error metrics.
- **Body Mask Constraint:** All 2D metrics strictly use a `get_body_mask()` function. Metrics evaluated on the surrounding "Air" (table/background) artificially inflate scores. PG-MambaGAN only evaluates metrics inside the patient's anatomical volume.

## 2. 3D Volumetric Assembly & NIfTI
Using preserved DICOM metadata, 2D predictions are stacked in precise Z-order to reconstruct a 3D volume.
- Output format: `.nii.gz` (NIfTI format).

## 3. 3D Metrics
Evaluated on the full `(D, H, W)` volumes.
- **3D-SSIM:** Structural similarity in 3 dimensions.
- **Flickering Index:** Measures Z-axis temporal instability between adjacent slices. A high flickering index means the model is treating each slice totally independently, causing a "strobe" effect in sagittal/coronal views. Mamba helps reduce this.

## 4. Hallucination Risk Analysis (`pyradiomics`)
This pipeline evaluates the volume using PyRadiomics to extract radiomic features (GLCM, GLRLM, shape features) of tumors or regions of interest. 
- **Preservation Rate:** Checks if the generative model destroyed radiomic signatures used by oncologists.

## 5. Clinical Task Validation
- **EPI (Edge Preservation Index):** Evaluates if microcalcification boundaries or lung fissures are blurred.
- **CNR (Contrast-to-Noise Ratio):** Evaluates tumor detectability against surrounding tissue backgrounds. A successful model improves CNR over the LDCT baseline.
