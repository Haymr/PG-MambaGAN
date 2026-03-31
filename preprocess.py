"""
PG-MambaGAN — DICOM Preprocessing Pipeline

Converts raw DICOM data to preprocessed NPY files for training.
Supports Mayo Clinic and PhantomX dataset structures.

Updates from original:
    - 512×512 output (was 256×256)
    - Patient ID preserved in filenames
    - Patient manifest JSON generation
    - NIfTI export support
    - Platform-independent paths (pathlib)
    - Configurable HU window

Usage:
    python preprocess.py \\
        --input-dir D:\\data\\LDCT-and-Projection-data \\
        --output-dir D:\\data\\processed_512 \\
        --img-size 512 \\
        --create-manifest

    # PhantomX:
    python preprocess.py \\
        --input-dir /data/phantomx \\
        --output-dir /data/phantomx_processed \\
        --img-size 512 \\
        --dataset-type phantomx
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np

try:
    import pydicom
except ImportError:
    pydicom = None
    print("⚠️  pydicom not installed. Install: pip install pydicom")

try:
    import cv2
except ImportError:
    cv2 = None
    print("⚠️  opencv not installed. Install: pip install opencv-python")

try:
    import nibabel as nib
except ImportError:
    nib = None


# ======================================================================
# Constants
# ======================================================================

DEFAULT_HU_MIN = -1000
DEFAULT_HU_MAX = 1000
DEFAULT_IMG_SIZE = 512


# ======================================================================
# Core Processing Functions
# ======================================================================

def dicom_to_hu(dcm) -> np.ndarray:
    """Convert DICOM pixel array to Hounsfield Units."""
    pixel_array = dcm.pixel_array.astype(np.float32)
    
    intercept = getattr(dcm, "RescaleIntercept", 0)
    slope = getattr(dcm, "RescaleSlope", 1)
    
    return pixel_array * float(slope) + float(intercept)


def normalize_hu(
    image: np.ndarray,
    hu_min: int = DEFAULT_HU_MIN,
    hu_max: int = DEFAULT_HU_MAX,
) -> np.ndarray:
    """
    Normalize HU values to [-1, 1] range.
    
    Pipeline:
        1. Clip to [hu_min, hu_max]
        2. Scale to [0, 1]
        3. Scale to [-1, 1]
    """
    image = np.clip(image, hu_min, hu_max)
    image = (image - hu_min) / (hu_max - hu_min)  # [0, 1]
    image = image * 2 - 1  # [-1, 1]
    return image.astype(np.float32)


def resize_image(image: np.ndarray, size: int = DEFAULT_IMG_SIZE) -> np.ndarray:
    """Resize image to square dimensions using area interpolation."""
    if cv2 is None:
        raise ImportError("opencv-python required for resizing")
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def extract_dicom_metadata(dcm) -> Dict:
    """
    Extract spatial metadata from a DICOM file for NIfTI reconstruction.
    
    Preserves the anatomical coordinate information needed to reconstruct
    3D volumes with correct geometry after 2D inference.
    
    Returns:
        Dict with PixelSpacing, SliceThickness, ImagePosition, etc.
    """
    meta = {}
    
    # Pixel spacing (in-plane resolution, mm)
    ps = getattr(dcm, "PixelSpacing", [1.0, 1.0])
    meta["pixel_spacing"] = [float(ps[0]), float(ps[1])]
    
    # Slice thickness (z-axis resolution, mm)
    meta["slice_thickness"] = float(getattr(dcm, "SliceThickness", 1.0))
    
    # Spacing between slices (may differ from thickness)
    meta["spacing_between_slices"] = float(
        getattr(dcm, "SpacingBetweenSlices", meta["slice_thickness"])
    )
    
    # Image position (patient) — origin of the volume
    pos = getattr(dcm, "ImagePositionPatient", [0.0, 0.0, 0.0])
    meta["image_position"] = [float(p) for p in pos]
    
    # Image orientation (patient) — direction cosines
    orient = getattr(dcm, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0])
    meta["image_orientation"] = [float(o) for o in orient]
    
    # Original image dimensions
    meta["rows"] = int(getattr(dcm, "Rows", 512))
    meta["columns"] = int(getattr(dcm, "Columns", 512))
    
    # Acquisition parameters (for provenance)
    meta["kvp"] = float(getattr(dcm, "KVP", 0))
    meta["exposure"] = float(getattr(dcm, "Exposure", 0))
    meta["manufacturer"] = str(getattr(dcm, "Manufacturer", "unknown"))
    
    return meta


# ======================================================================
# Mayo Clinic Dataset Processing
# ======================================================================

def find_dose_folders(patient_path: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Find Low Dose and Full Dose folders in a Mayo Clinic patient directory.
    Skips projection/sinogram folders.
    """
    low_path = None
    high_path = None
    
    for root, dirs, _ in os.walk(str(patient_path)):
        for d in dirs:
            d_lower = d.lower()
            
            # Skip projections
            if "proj" in d_lower or "sino" in d_lower:
                continue
            
            if "low dose" in d_lower:
                low_path = Path(root) / d
            elif "full dose" in d_lower or "high dose" in d_lower:
                high_path = Path(root) / d
    
    return low_path, high_path


def process_mayo(
    input_dir: Path,
    output_dir: Path,
    img_size: int = DEFAULT_IMG_SIZE,
    hu_min: int = DEFAULT_HU_MIN,
    hu_max: int = DEFAULT_HU_MAX,
) -> dict:
    """
    Process Mayo Clinic LDCT dataset.
    
    Structure:
        input_dir/C001/... → Low Dose / Full Dose
        input_dir/L004/... → Low Dose / Full Dose
    
    Returns:
        dict mapping patient_id -> list of output filenames
    """
    (output_dir / "trainA").mkdir(parents=True, exist_ok=True)
    (output_dir / "trainB").mkdir(parents=True, exist_ok=True)
    
    # Find patient directories (C and L prefixes)
    patient_dirs = sorted([
        d for d in input_dir.iterdir()
        if d.is_dir() and d.name[0] in ("C", "L")
    ])
    
    if not patient_dirs:
        raise FileNotFoundError(
            f"No patient directories (C*/L*) found in {input_dir}"
        )
    
    print(f"Found {len(patient_dirs)} patients")
    
    patient_files = {}
    total_processed = 0
    
    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        low_path, high_path = find_dose_folders(patient_dir)
        
        if not low_path or not high_path:
            print(f"  ⚠️  {patient_id}: Low/Full dose folders not found — skipping")
            continue
        
        print(f"  Processing {patient_id}...")
        
        low_files = sorted([f for f in low_path.iterdir() if f.suffix == ".dcm"])
        high_files = sorted([f for f in high_path.iterdir() if f.suffix == ".dcm"])
        
        n_slices = min(len(low_files), len(high_files))
        saved_files = []
        
        for i in range(n_slices):
            try:
                low_dcm = pydicom.dcmread(str(low_files[i]))
                high_dcm = pydicom.dcmread(str(high_files[i]))
                
                # HU conversion
                low_hu = dicom_to_hu(low_dcm)
                high_hu = dicom_to_hu(high_dcm)
                
                # Clip extreme values before resizing
                low_hu = np.clip(low_hu, hu_min, hu_max)
                high_hu = np.clip(high_hu, hu_min, hu_max)

                # Resize
                low_img = resize_image(low_hu, img_size)
                high_img = resize_image(high_hu, img_size)
                
                # Normalize
                low_norm = normalize_hu(low_img, hu_min, hu_max)
                high_norm = normalize_hu(high_img, hu_min, hu_max)
                
                # Save — filename preserves patient ID
                filename = f"{patient_id}_{i:04d}.npy"
                np.save(output_dir / "trainA" / filename, low_norm)
                np.save(output_dir / "trainB" / filename, high_norm)
                
                saved_files.append(filename)
                total_processed += 1
                
            except Exception as e:
                print(f"    Error ({patient_id} slice {i}): {e}")
                continue
        
        if saved_files:
            patient_files[patient_id] = saved_files
            print(f"    ✅ {len(saved_files)} slices processed")
            
            # Save DICOM spatial metadata for NIfTI reconstruction
            try:
                first_dcm = pydicom.dcmread(str(high_files[0]))
                patient_meta = extract_dicom_metadata(first_dcm)
                patient_meta["n_slices"] = len(saved_files)
                patient_meta["patient_id"] = patient_id
                patient_meta["img_size_processed"] = img_size
                patient_meta["hu_range"] = [hu_min, hu_max]
                
                # Collect all slice positions for accurate z-ordering
                slice_positions = []
                for f in high_files[:n_slices]:
                    d = pydicom.dcmread(str(f), stop_before_pixels=True)
                    pos = getattr(d, "ImagePositionPatient", [0, 0, 0])
                    slice_positions.append(float(pos[2]))
                patient_meta["slice_positions_z"] = slice_positions
                
                meta_dir = output_dir / "metadata"
                meta_dir.mkdir(exist_ok=True)
                with open(meta_dir / f"{patient_id}_meta.json", "w") as mf:
                    json.dump(patient_meta, mf, indent=2)
            except Exception as e:
                print(f"    ⚠️  Metadata extraction failed: {e}")
    
    print(f"\n✅ Total: {total_processed} slices from {len(patient_files)} patients")
    return patient_files


# ======================================================================
# PhantomX Dataset Processing
# ======================================================================

def process_phantomx(
    input_dir: Path,
    output_dir: Path,
    img_size: int = DEFAULT_IMG_SIZE,
    hu_min: int = DEFAULT_HU_MIN,
    hu_max: int = DEFAULT_HU_MAX,
    recon_type: str = "FBP",
) -> dict:
    """
    Process PhantomX dataset.
    
    Structure:
        input_dir/D55-01/40/FBP/*.dcm  (low dose)
        input_dir/D55-01/300/FBP/*.dcm (high dose)
    """
    (output_dir / "trainA").mkdir(parents=True, exist_ok=True)
    (output_dir / "trainB").mkdir(parents=True, exist_ok=True)
    
    phantoms = sorted([
        d for d in input_dir.iterdir()
        if d.is_dir() and d.name.startswith("D")
    ])
    
    print(f"Found {len(phantoms)} phantoms")
    
    patient_files = {}
    
    for phantom_dir in phantoms:
        phantom_id = phantom_dir.name
        
        # Find low (40 mAs) and high (300 mAs) dose + reconstruction
        low_dose_path = None
        high_dose_path = None
        
        for dose_dir in phantom_dir.iterdir():
            if not dose_dir.is_dir():
                continue
            
            for recon_dir in dose_dir.iterdir():
                if recon_type.upper() in recon_dir.name.upper():
                    if dose_dir.name == "40":
                        low_dose_path = recon_dir
                    elif dose_dir.name == "300":
                        high_dose_path = recon_dir
        
        if not low_dose_path or not high_dose_path:
            print(f"  ⚠️  {phantom_id}: Dose folders not found — skipping")
            continue
        
        print(f"  Processing {phantom_id} ({recon_type})...")
        
        low_files = sorted([f for f in low_dose_path.iterdir() if f.suffix == ".dcm"])
        high_files = sorted([f for f in high_dose_path.iterdir() if f.suffix == ".dcm"])
        
        n_slices = min(len(low_files), len(high_files))
        saved_files = []
        
        for i in range(n_slices):
            try:
                low_dcm = pydicom.dcmread(str(low_files[i]))
                high_dcm = pydicom.dcmread(str(high_files[i]))
                
                low_hu = dicom_to_hu(low_dcm)
                high_hu = dicom_to_hu(high_dcm)
                
                low_hu = np.clip(low_hu, hu_min, hu_max)
                high_hu = np.clip(high_hu, hu_min, hu_max)

                low_img = resize_image(low_hu, img_size)
                high_img = resize_image(high_hu, img_size)
                
                low_norm = normalize_hu(low_img, hu_min, hu_max)
                high_norm = normalize_hu(high_img, hu_min, hu_max)
                
                filename = f"{phantom_id}_{i:04d}.npy"
                np.save(output_dir / "trainA" / filename, low_norm)
                np.save(output_dir / "trainB" / filename, high_norm)
                
                saved_files.append(filename)
                
            except Exception as e:
                print(f"    Error ({phantom_id} slice {i}): {e}")
                continue
        
        if saved_files:
            patient_files[phantom_id] = saved_files
            print(f"    ✅ {len(saved_files)} slices processed")
    
    return patient_files


# ======================================================================
# Manifest Generation
# ======================================================================

def create_manifest_from_processed(
    output_dir: Path,
    patient_files: dict,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """Create a patient manifest from processed data."""
    from data.patient_manifest import PatientManifest
    
    # Build manifest from known patient files
    manifest = PatientManifest.from_flat_npy(
        data_root=str(output_dir),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    
    # Verify no leakage
    manifest.verify_no_leakage()
    
    # Save
    manifest.save(str(output_dir / "patient_manifest.json"))


# ======================================================================
# CLI Entry Point
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="PG-MambaGAN DICOM Preprocessing Pipeline"
    )
    
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Path to raw DICOM dataset"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Path to save processed NPY files"
    )
    parser.add_argument(
        "--img-size", type=int, default=DEFAULT_IMG_SIZE,
        help=f"Output image size (default: {DEFAULT_IMG_SIZE})"
    )
    parser.add_argument(
        "--hu-min", type=int, default=DEFAULT_HU_MIN,
        help=f"Minimum HU value for clipping (default: {DEFAULT_HU_MIN})"
    )
    parser.add_argument(
        "--hu-max", type=int, default=DEFAULT_HU_MAX,
        help=f"Maximum HU value for clipping (default: {DEFAULT_HU_MAX})"
    )
    parser.add_argument(
        "--dataset-type", type=str, default="mayo",
        choices=["mayo", "phantomx"],
        help="Dataset type (default: mayo)"
    )
    parser.add_argument(
        "--recon-type", type=str, default="FBP",
        help="PhantomX reconstruction type (default: FBP)"
    )
    parser.add_argument(
        "--create-manifest", action="store_true",
        help="Create patient manifest after processing"
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8,
        help="Train split ratio (default: 0.8)"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1,
        help="Validation split ratio (default: 0.1)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for splitting (default: 42)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    print("=" * 60)
    print("  PG-MambaGAN Preprocessing Pipeline")
    print("=" * 60)
    print(f"  Input:       {input_dir}")
    print(f"  Output:      {output_dir}")
    print(f"  Image Size:  {args.img_size}×{args.img_size}")
    print(f"  HU Range:    [{args.hu_min}, {args.hu_max}]")
    print(f"  Dataset:     {args.dataset_type}")
    print("=" * 60)
    
    # Check prerequisites
    if pydicom is None:
        raise ImportError("pydicom required: pip install pydicom")
    if cv2 is None:
        raise ImportError("opencv required: pip install opencv-python")
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Process
    if args.dataset_type == "mayo":
        patient_files = process_mayo(
            input_dir, output_dir, args.img_size,
            args.hu_min, args.hu_max
        )
    elif args.dataset_type == "phantomx":
        patient_files = process_phantomx(
            input_dir, output_dir, args.img_size,
            args.hu_min, args.hu_max, args.recon_type
        )
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")
    
    # Create manifest
    if args.create_manifest and patient_files:
        test_ratio = 1.0 - args.train_ratio - args.val_ratio
        print(f"\nCreating patient manifest...")
        create_manifest_from_processed(
            output_dir, patient_files,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=test_ratio,
            seed=args.seed,
        )
    
    print("\n✅ Preprocessing complete!")


if __name__ == "__main__":
    main()
