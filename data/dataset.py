"""
PG-MambaGAN — LDCT Dataset (PyTorch + MONAI)

Flexible data pipeline supporting multiple dataset structures:
    - flat: trainA/trainB NPY files (preprocessed)
    - patient_folders: DICOM patient directories
    - paired_nifti: NIfTI file pairs
    - custom: user-defined manifest.json

Usage:
    from data.dataset import LDCTDataset
    from data.patient_manifest import PatientManifest
    
    manifest = PatientManifest.load("data/mayo/patient_manifest.json")
    
    dataset = LDCTDataset(
        manifest=manifest,
        split="train",
        data_root="data/mayo/processed",
        img_size=512,
        augment=True,
    )
    
    dataloader = dataset.get_dataloader(batch_size=1, num_workers=4)
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from data.patient_manifest import PatientManifest


class LDCTDataset(Dataset):
    """
    Low-Dose CT denoising dataset with patient-level split support.
    
    Loads paired (LDCT, NDCT) images and applies optional augmentations.
    Currently supports the flat NPY structure (primary use case).
    
    Args:
        manifest: PatientManifest instance with patient-level splits.
        split: One of "train", "val", "test".
        data_root: Path to the preprocessed data directory.
        img_size: Expected image size (square). Used for validation.
        augment: Whether to apply data augmentation (only for training).
        subdirs: Tuple of (input_subdir, target_subdir) directory names.
    """
    
    def __init__(
        self,
        manifest: PatientManifest,
        split: str = "train",
        data_root: Optional[str] = None,
        img_size: int = 512,
        augment: bool = False,
        subdirs: Tuple[str, str] = ("trainA", "trainB"),
    ):
        super().__init__()
        
        self.manifest = manifest
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == "train")
        
        # Get file lists from manifest
        self.input_files, self.target_files = manifest.get_split_files(
            split=split,
            data_root=data_root,
            subdirs=subdirs,
        )
        
        if len(self.input_files) == 0:
            raise ValueError(
                f"No files found for split '{split}'. "
                f"Check data_root and manifest."
            )
        
        print(f"  LDCTDataset [{split}]: {len(self.input_files)} paired images"
              f"{' (augmented)' if self.augment else ''}")
    
    def __len__(self) -> int:
        return len(self.input_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load a paired (LDCT, NDCT) sample.
        
        Returns:
            Dict with keys:
                - "ldct": Low-dose CT tensor (1, H, W)
                - "ndct": Normal-dose CT tensor (1, H, W)
                - "filename": Source filename string
        """
        # Load NPY files
        ldct = np.load(self.input_files[idx]).astype(np.float32)
        ndct = np.load(self.target_files[idx]).astype(np.float32)
        
        # Ensure 2D → (H, W) — remove extra dims if present
        if ldct.ndim == 3:
            ldct = ldct[:, :, 0] if ldct.shape[-1] == 1 else ldct[0]
        if ndct.ndim == 3:
            ndct = ndct[:, :, 0] if ndct.shape[-1] == 1 else ndct[0]
        
        # Apply augmentations
        if self.augment:
            ldct, ndct = self._augment(ldct, ndct)
        
        # Convert to tensors: (H, W) → (1, H, W)
        ldct_tensor = torch.from_numpy(ldct).unsqueeze(0)
        ndct_tensor = torch.from_numpy(ndct).unsqueeze(0)
        
        return {
            "ldct": ldct_tensor,
            "ndct": ndct_tensor,
            "filename": self.input_files[idx].name,
        }
    
    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------
    
    def _augment(
        self, ldct: np.ndarray, ndct: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply paired augmentations to both LDCT and NDCT images.
        
        Augmentations (applied identically to both images):
            - Random horizontal flip (p=0.5)
            - Random vertical flip (p=0.5)
            - Random rotation: 0, 90, 180, or 270 degrees (p=0.5)
        
        Note: We use simple geometric augmentations that preserve HU values.
        Intensity augmentations are NOT used because they would corrupt
        the HU-based tissue segmentation in the NPS loss.
        """
        # Horizontal flip
        if np.random.random() > 0.5:
            ldct = np.flip(ldct, axis=1).copy()
            ndct = np.flip(ndct, axis=1).copy()
        
        # Vertical flip
        if np.random.random() > 0.5:
            ldct = np.flip(ldct, axis=0).copy()
            ndct = np.flip(ndct, axis=0).copy()
        
        # Random 90° rotation
        if np.random.random() > 0.5:
            k = np.random.choice([1, 2, 3])
            ldct = np.rot90(ldct, k=k).copy()
            ndct = np.rot90(ndct, k=k).copy()
        
        return ldct, ndct
    
    # ------------------------------------------------------------------
    # DataLoader Factory
    # ------------------------------------------------------------------
    
    def get_dataloader(
        self,
        batch_size: int = 1,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = True,
    ) -> DataLoader:
        """
        Create a DataLoader with optimal settings.
        
        Args:
            batch_size: Batch size (physical, pre-accumulation).
            num_workers: Number of data loading workers.
            pin_memory: Pin memory for faster GPU transfer.
            drop_last: Drop the last incomplete batch.
            
        Returns:
            PyTorch DataLoader instance.
        """
        shuffle = (self.split == "train")
        
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            drop_last=drop_last and (self.split == "train"),
            persistent_workers=num_workers > 0,
        )
    
    # ------------------------------------------------------------------
    # Utility: Get full volume for a patient (3D evaluation)
    # ------------------------------------------------------------------
    
    def get_patient_volume(
        self, patient_id: str, data_root: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load ALL slices for a specific patient as a 3D volume.
        
        Used for 3D volumetric evaluation (FAZ 5 — Revizyon 3).
        
        Args:
            patient_id: Patient identifier.
            data_root: Override data root path.
            
        Returns:
            Tuple of (ldct_volume, ndct_volume) as (N, H, W) arrays.
        """
        root = Path(data_root or self.manifest.metadata.get("data_root", "."))
        
        if patient_id not in self.manifest.patients:
            raise ValueError(f"Patient '{patient_id}' not found in manifest.")
        
        filenames = sorted(self.manifest.patients[patient_id])
        
        ldct_slices = []
        ndct_slices = []
        
        for fname in filenames:
            ldct_path = root / "trainA" / fname
            ndct_path = root / "trainB" / fname
            
            if ldct_path.exists() and ndct_path.exists():
                ldct = np.load(ldct_path).astype(np.float32)
                ndct = np.load(ndct_path).astype(np.float32)
                
                if ldct.ndim == 3:
                    ldct = ldct[:, :, 0] if ldct.shape[-1] == 1 else ldct[0]
                if ndct.ndim == 3:
                    ndct = ndct[:, :, 0] if ndct.shape[-1] == 1 else ndct[0]
                
                ldct_slices.append(ldct)
                ndct_slices.append(ndct)
        
        if not ldct_slices:
            raise ValueError(f"No slice files found for patient '{patient_id}'")
        
        return np.stack(ldct_slices), np.stack(ndct_slices)
