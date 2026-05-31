"""
PG-MambaGAN — 3D Volumetric Assembly & NIfTI Export

Stacks 2D generator outputs back into 3D NIfTI volumes with
DICOM-derived spatial metadata preservation.

Key design:
    DICOM metadata (PixelSpacing, SliceThickness, ImagePosition,
    ImageOrientation) is saved during preprocessing as per-patient
    JSON files. This module reads that metadata and constructs
    proper NIfTI affine matrices so that resulting volumes are
    anatomically correct in radiological viewers.

Pipeline:
    1. Load per-patient metadata JSON
    2. Infer 2D slices through generator
    3. Stack into (N, H, W) volume
    4. Build NIfTI affine from DICOM metadata
    5. Export as .nii.gz with correct geometry

Usage:
    assembler = VolumeAssembler(
        generator=G_ema,
        metadata_dir="data/mayo/processed/metadata",
        output_dir="experiments/volumes",
    )
    assembler.assemble_patient("C001", dataset, device="cuda")
"""

import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    print("⚠️  nibabel not installed. NIfTI export disabled.")


# ======================================================================
# Affine Matrix Construction
# ======================================================================

def build_affine_from_dicom_meta(meta: Dict) -> np.ndarray:
    """
    Construct a 4×4 NIfTI affine matrix from DICOM metadata.
    
    The affine maps voxel indices (i, j, k) to patient coordinates (x, y, z).
    
    Uses:
        - ImageOrientationPatient → row/column direction cosines
        - PixelSpacing → in-plane resolution
        - SliceThickness / SpacingBetweenSlices → z-resolution
        - ImagePositionPatient → volume origin
    
    Args:
        meta: Dict from {patient_id}_meta.json
    
    Returns:
        4×4 affine matrix (numpy array)
    """
    # Direction cosines
    orient = meta.get("image_orientation", [1, 0, 0, 0, 1, 0])
    row_cosine = np.array(orient[:3], dtype=np.float64)
    col_cosine = np.array(orient[3:6], dtype=np.float64)
    
    # Slice direction (cross product of row × col)
    slice_cosine = np.cross(row_cosine, col_cosine)
    
    # Pixel spacing
    ps = meta.get("pixel_spacing", [1.0, 1.0])
    dx, dy = float(ps[0]), float(ps[1])
    
    # Slice spacing (prefer SpacingBetweenSlices, fallback to SliceThickness)
    dz = float(meta.get("spacing_between_slices",
                         meta.get("slice_thickness", 1.0)))
    
    # Handle preprocessed image size (may differ from original)
    original_rows = meta.get("rows", 512)
    processed_size = meta.get("img_size_processed", 512)
    if original_rows != processed_size and original_rows > 0:
        scale = original_rows / processed_size
        dx *= scale
        dy *= scale
    
    # Origin
    origin = meta.get("image_position", [0.0, 0.0, 0.0])
    
    # Build 4×4 affine
    affine = np.eye(4, dtype=np.float64)
    
    affine[0, 0] = row_cosine[0] * dx
    affine[0, 1] = col_cosine[0] * dy
    affine[0, 2] = slice_cosine[0] * dz
    affine[0, 3] = float(origin[0])
    
    affine[1, 0] = row_cosine[1] * dx
    affine[1, 1] = col_cosine[1] * dy
    affine[1, 2] = slice_cosine[1] * dz
    affine[1, 3] = float(origin[1])
    
    affine[2, 0] = row_cosine[2] * dx
    affine[2, 1] = col_cosine[2] * dy
    affine[2, 2] = slice_cosine[2] * dz
    affine[2, 3] = float(origin[2])
    
    return affine


# ======================================================================
# Volume Assembler
# ======================================================================

class VolumeAssembler:
    """
    Assembles 2D generator outputs into 3D NIfTI volumes.
    
    Preserves DICOM spatial metadata for anatomically correct volumes.
    
    Args:
        generator: Generator model (should be in eval mode, e.g., G_ema).
        metadata_dir: Path to directory with {patient_id}_meta.json files.
        output_dir: Path to save output NIfTI files.
        hu_min: HU minimum for denormalization.
        hu_max: HU maximum for denormalization.
    """
    
    def __init__(
        self,
        generator: nn.Module,
        metadata_dir: Optional[str] = None,
        output_dir: str = "experiments/volumes",
        hu_min: int = -1000,
        hu_max: int = 1000,
    ):
        self.generator = generator
        self.metadata_dir = Path(metadata_dir) if metadata_dir else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.hu_min = hu_min
        self.hu_max = hu_max
    
    def _load_patient_metadata(self, patient_id: str) -> Optional[Dict]:
        """Load DICOM metadata JSON for a patient."""
        if self.metadata_dir is None:
            return None
        
        meta_path = self.metadata_dir / f"{patient_id}_meta.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                return json.load(f)
        return None
    
    def _denormalize_to_hu(self, image: np.ndarray) -> np.ndarray:
        """Convert [-1, 1] → HU values."""
        return np.clip((image + 1.0) / 2.0 * (self.hu_max - self.hu_min) + self.hu_min, self.hu_min, self.hu_max)
    
    @torch.no_grad()
    def assemble_patient(
        self,
        patient_id: str,
        dataset,
        device: str = "cuda",
        save_hu: bool = True,
        use_amp: bool = True,
    ) -> Dict:
        """
        Assemble a complete 3D volume for one patient.
        
        Steps:
            1. Get ALL slices for this patient from the dataset
            2. Infer through generator (slice by slice)
            3. Stack into 3D volume (N, H, W)
            4. Build NIfTI with DICOM-derived affine
            5. Save as .nii.gz
        
        Args:
            patient_id: Patient identifier.
            dataset: LDCTDataset with the patient's data.
            device: Compute device.
            save_hu: If True, save in HU values. If False, save normalized.
            use_amp: Use mixed precision for inference.
        
        Returns:
            Dict with volume paths and metadata.
        """
        self.generator.eval()
        
        # Load all slices for this patient
        data_root = dataset.manifest.metadata.get("data_root", ".")
        ldct_vol, ndct_vol = dataset.get_patient_volume(patient_id, data_root)
        
        n_slices = ldct_vol.shape[0]
        print(f"  Assembling {patient_id}: {n_slices} slices...")
        
        # Infer slice by slice
        pred_slices = []
        
        for i in range(n_slices):
            ldct_slice = torch.from_numpy(ldct_vol[i]).unsqueeze(0).unsqueeze(0)
            ldct_slice = ldct_slice.to(device)
            
            if use_amp and device == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred = self.generator(ldct_slice)
            else:
                pred = self.generator(ldct_slice)
            
            pred_slices.append(pred.float().squeeze().cpu().numpy())
        
        pred_vol = np.stack(pred_slices)  # (N, H, W)
        
        # Load DICOM metadata
        meta = self._load_patient_metadata(patient_id)
        
        # Sort by z-position if available
        if meta and "slice_positions_z" in meta:
            z_positions = meta["slice_positions_z"]
            if len(z_positions) == n_slices:
                sort_idx = np.argsort(z_positions)
                pred_vol = pred_vol[sort_idx]
                ndct_vol = ndct_vol[sort_idx]
                ldct_vol = ldct_vol[sort_idx]
        
        # Optionally convert to HU
        if save_hu:
            pred_vol_save = self._denormalize_to_hu(pred_vol)
            ndct_vol_save = self._denormalize_to_hu(ndct_vol)
            ldct_vol_save = self._denormalize_to_hu(ldct_vol)
        else:
            pred_vol_save = pred_vol
            ndct_vol_save = ndct_vol
            ldct_vol_save = ldct_vol
        
        # Build affine matrix
        if meta:
            affine = build_affine_from_dicom_meta(meta)
        else:
            # Fallback: identity with 1mm isotropic spacing
            affine = np.eye(4, dtype=np.float64)
            print(f"  ⚠️  No metadata found for {patient_id}. Using identity affine.")
        
        # Save NIfTI volumes
        result = {}
        patient_dir = self.output_dir / patient_id
        patient_dir.mkdir(exist_ok=True)
        
        if NIBABEL_AVAILABLE:
            for name, vol in [
                ("predicted", pred_vol_save),
                ("ndct", ndct_vol_save),
                ("ldct", ldct_vol_save),
            ]:
                # NIfTI expects (X, Y, Z), we have (Z, Y, X) → transpose
                nifti_vol = np.transpose(vol, (2, 1, 0))
                img = nib.Nifti1Image(nifti_vol.astype(np.float32), affine)
                
                out_path = patient_dir / f"{name}.nii.gz"
                nib.save(img, str(out_path))
                result[f"{name}_path"] = str(out_path)
            
            print(f"  ✅ NIfTI volumes saved: {patient_dir}")
        else:
            # Fallback: save as NPY
            for name, vol in [("predicted", pred_vol_save),
                              ("ndct", ndct_vol_save)]:
                np.save(patient_dir / f"{name}.npy", vol)
                result[f"{name}_path"] = str(patient_dir / f"{name}.npy")
            
            print(f"  ✅ NPY volumes saved (nibabel not available): {patient_dir}")
        
        # Return volumes in normalized space for metric computation
        result["pred_volume"] = pred_vol
        result["ndct_volume"] = ndct_vol
        result["ldct_volume"] = ldct_vol
        result["n_slices"] = n_slices
        result["patient_id"] = patient_id
        result["affine"] = affine
        
        return result
