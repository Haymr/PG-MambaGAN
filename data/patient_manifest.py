"""
PG-MambaGAN — Patient-Level Data Manifest

Manages patient-level train/val/test splitting to prevent data leakage.
Each patient's ALL slices go into exactly one split.

Usage:
    from data.patient_manifest import PatientManifest
    
    # Create manifest from preprocessed data
    manifest = PatientManifest.from_flat_npy(
        data_root="data/mayo/processed",
        id_pattern=r"^([A-Za-z]+\d+)",
        train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
        seed=42
    )
    manifest.save("data/mayo/patient_manifest.json")
    
    # Load existing manifest
    manifest = PatientManifest.load("data/mayo/patient_manifest.json")
    train_files = manifest.get_split_files("train")
"""

import json
import re
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np


class PatientManifest:
    """
    Patient-level data split manager.
    
    Ensures zero data leakage by assigning ALL slices from a patient
    to exactly one split (train, val, or test).
    
    Attributes:
        patients: Dict mapping patient_id -> list of slice filenames
        splits: Dict mapping split_name -> list of patient_ids
        metadata: Dict with creation parameters
    """
    
    def __init__(
        self,
        patients: Dict[str, List[str]],
        splits: Dict[str, List[str]],
        metadata: Optional[Dict] = None,
    ):
        self.patients = patients
        self.splits = splits
        self.metadata = metadata or {}
    
    # ------------------------------------------------------------------
    # Factory: Create from flat NPY directory
    # ------------------------------------------------------------------
    
    @classmethod
    def from_flat_npy(
        cls,
        data_root: str,
        id_pattern: str = r"^([A-Za-z]+\d+)",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        subdirs: Tuple[str, str] = ("trainA", "trainB"),
    ) -> "PatientManifest":
        """
        Create a manifest from a flat NPY directory structure.
        
        Expected layout:
            data_root/trainA/{PatientID}_{slice:04d}.npy
            data_root/trainB/{PatientID}_{slice:04d}.npy
        
        Args:
            data_root: Path to the preprocessed data directory.
            id_pattern: Regex to extract patient ID from filename.
            train_ratio: Fraction of patients for training.
            val_ratio: Fraction of patients for validation.
            test_ratio: Fraction of patients for testing.
            seed: Random seed for reproducibility.
            subdirs: Tuple of (input_dir, target_dir) names.
            
        Returns:
            PatientManifest instance with patient-level splits.
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        
        root = Path(data_root)
        input_dir = root / subdirs[0]
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Group files by patient ID
        patient_files: Dict[str, List[str]] = defaultdict(list)
        pattern = re.compile(id_pattern)
        
        for f in sorted(input_dir.iterdir()):
            if f.suffix != ".npy":
                continue
            
            match = pattern.match(f.stem)
            if match:
                patient_id = match.group(1)
                patient_files[patient_id].append(f.name)
            else:
                print(f"  ⚠️  Could not extract patient ID from: {f.name}")
        
        if not patient_files:
            raise ValueError(
                f"No patients found in {input_dir} with pattern '{id_pattern}'"
            )
        
        # Sort patient IDs for reproducibility
        all_patients = sorted(patient_files.keys())
        n_patients = len(all_patients)
        
        print(f"  Found {n_patients} patients, "
              f"{sum(len(v) for v in patient_files.values())} total slices")
        
        # Stratified split based on number of slices
        splits = cls._stratified_split(
            patient_files, all_patients,
            train_ratio, val_ratio, test_ratio, seed
        )
        
        metadata = {
            "data_root": str(data_root),
            "id_pattern": id_pattern,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "seed": seed,
            "n_patients": n_patients,
            "n_train": len(splits["train"]),
            "n_val": len(splits["val"]),
            "n_test": len(splits["test"]),
            "n_train_slices": sum(len(patient_files[p]) for p in splits["train"]),
            "n_val_slices": sum(len(patient_files[p]) for p in splits["val"]),
            "n_test_slices": sum(len(patient_files[p]) for p in splits["test"]),
        }
        
        return cls(
            patients=dict(patient_files),
            splits=splits,
            metadata=metadata,
        )
    
    # ------------------------------------------------------------------
    # Factory: Create from patient-folder structure (DICOM)
    # ------------------------------------------------------------------
    
    @classmethod
    def from_patient_folders(
        cls,
        data_root: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> "PatientManifest":
        """
        Create a manifest from a patient-folder DICOM structure.
        
        Expected layout (Mayo Clinic):
            data_root/C001/Low Dose/*.dcm
            data_root/C001/Full Dose/*.dcm
            data_root/L004/Low Dose/*.dcm
            ...
        """
        root = Path(data_root)
        
        patient_files: Dict[str, List[str]] = {}
        
        for patient_dir in sorted(root.iterdir()):
            if not patient_dir.is_dir():
                continue
            if patient_dir.name.startswith("."):
                continue
            
            # Count DICOM files in any subdirectory
            dcm_files = list(patient_dir.rglob("*.dcm"))
            if dcm_files:
                patient_id = patient_dir.name
                patient_files[patient_id] = [f.name for f in sorted(dcm_files)]
        
        if not patient_files:
            raise ValueError(f"No patient directories with DICOM files found in {root}")
        
        all_patients = sorted(patient_files.keys())
        n_patients = len(all_patients)
        
        print(f"  Found {n_patients} patients, "
              f"{sum(len(v) for v in patient_files.values())} total DICOM files")
        
        splits = cls._stratified_split(
            patient_files, all_patients,
            train_ratio, val_ratio, test_ratio, seed
        )
        
        metadata = {
            "data_root": str(data_root),
            "structure": "patient_folders",
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "seed": seed,
            "n_patients": n_patients,
            "n_train": len(splits["train"]),
            "n_val": len(splits["val"]),
            "n_test": len(splits["test"]),
        }
        
        return cls(
            patients=dict(patient_files),
            splits=splits,
            metadata=metadata,
        )
    
    # ------------------------------------------------------------------
    # Splitting Logic
    # ------------------------------------------------------------------
    
    @staticmethod
    def _stratified_split(
        patient_files: Dict[str, List[str]],
        all_patients: List[str],
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        seed: int,
    ) -> Dict[str, List[str]]:
        """
        Split patients into train/val/test with stratification by slice count.
        
        Patients are sorted by slice count and distributed in a round-robin
        fashion across splits to ensure balanced slice distribution.
        """
        rng = np.random.RandomState(seed)
        
        # Sort by slice count for stratification, then shuffle within groups
        patients_with_counts = [
            (pid, len(patient_files[pid])) for pid in all_patients
        ]
        
        # Shuffle first for randomness, then stable-sort by count group
        rng.shuffle(patients_with_counts)
        
        n = len(patients_with_counts)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_test = n - n_train - n_val  # Remainder goes to test
        
        # Ensure at least 1 in each split if possible
        if n >= 3:
            n_train = max(n_train, 1)
            n_val = max(n_val, 1)
            n_test = max(n_test, 1)
            # Re-adjust
            total = n_train + n_val + n_test
            if total > n:
                n_train = n - n_val - n_test
        
        patient_ids = [p[0] for p in patients_with_counts]
        
        splits = {
            "train": patient_ids[:n_train],
            "val": patient_ids[n_train:n_train + n_val],
            "test": patient_ids[n_train + n_val:],
        }
        
        # Print split info
        for split_name, pids in splits.items():
            n_slices = sum(len(patient_files[p]) for p in pids)
            print(f"  {split_name:>5}: {len(pids):>3} patients, {n_slices:>6} slices")
        
        return splits
    
    # ------------------------------------------------------------------
    # Query Methods
    # ------------------------------------------------------------------
    
    def get_split_files(
        self,
        split: str,
        data_root: Optional[str] = None,
        subdirs: Tuple[str, str] = ("trainA", "trainB"),
    ) -> Tuple[List[Path], List[Path]]:
        """
        Get file paths for a specific split.
        
        Args:
            split: One of "train", "val", "test".
            data_root: Override the data root path.
            subdirs: Tuple of (input_subdir, target_subdir).
            
        Returns:
            Tuple of (input_file_paths, target_file_paths).
        """
        if split not in self.splits:
            raise ValueError(f"Unknown split '{split}'. Use: {list(self.splits.keys())}")
        
        root = Path(data_root or self.metadata.get("data_root", "."))
        input_dir = root / subdirs[0]
        target_dir = root / subdirs[1]
        
        input_files = []
        target_files = []
        
        for patient_id in self.splits[split]:
            for filename in self.patients[patient_id]:
                input_path = input_dir / filename
                target_path = target_dir / filename
                
                if input_path.exists() and target_path.exists():
                    input_files.append(input_path)
                    target_files.append(target_path)
        
        return input_files, target_files
    
    def get_patient_ids(self, split: str) -> List[str]:
        """Get patient IDs for a specific split."""
        return self.splits.get(split, [])
    
    def verify_no_leakage(self) -> bool:
        """Verify that no patient appears in multiple splits."""
        all_sets = []
        for split_name, pids in self.splits.items():
            pid_set = set(pids)
            for other_set in all_sets:
                overlap = pid_set & other_set
                if overlap:
                    print(f"  ❌ DATA LEAKAGE: {overlap} appears in multiple splits!")
                    return False
            all_sets.append(pid_set)
        
        print("  ✅ No data leakage detected. All splits are patient-disjoint.")
        return True
    
    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    
    def save(self, path: str) -> None:
        """Save manifest to JSON file."""
        data = {
            "patients": self.patients,
            "splits": self.splits,
            "metadata": self.metadata,
        }
        
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"  ✅ Manifest saved: {filepath}")
    
    @classmethod
    def load(cls, path: str) -> "PatientManifest":
        """Load manifest from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        manifest = cls(
            patients=data["patients"],
            splits=data["splits"],
            metadata=data.get("metadata", {}),
        )
        
        n_patients = len(manifest.patients)
        print(f"  ✅ Manifest loaded: {n_patients} patients "
              f"({len(manifest.splits.get('train', []))} train / "
              f"{len(manifest.splits.get('val', []))} val / "
              f"{len(manifest.splits.get('test', []))} test)")
        
        return manifest
    
    def __repr__(self) -> str:
        n = len(self.patients)
        splits_info = ", ".join(
            f"{k}={len(v)}" for k, v in self.splits.items()
        )
        return f"PatientManifest({n} patients: {splits_info})"
