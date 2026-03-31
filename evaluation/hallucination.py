"""
PG-MambaGAN — Hallucination Risk Analysis

Tests whether the generator preserves radiological features or hallucinates
new structures. Uses pyradiomics to extract quantitative features and
compares them between NDCT and denoised images.

A well-trained denoiser should:
    - PRESERVE: First Order statistics (Mean, Median, Entropy)
    - PRESERVE: GLCM texture (Contrast, Correlation, Homogeneity)
    - REDUCE: Noise-related features (high-frequency energy)
    - NOT INTRODUCE: New structures not present in NDCT

Methodology:
    1. Extract First Order + GLCM features from NDCT and Predicted
    2. Compute relative error: |feature_pred - feature_ndct| / |feature_ndct|
    3. Flag features with > threshold% deviation as potential hallucination
    4. Report per-feature preservation scores

Usage:
    analyzer = HallucinationAnalyzer()
    report = analyzer.analyze_patient("C001", pred_vol, ndct_vol)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False

try:
    from radiomics import featureextractor
    RADIOMICS_AVAILABLE = True
except ImportError:
    RADIOMICS_AVAILABLE = False


# ======================================================================
# Feature Extraction Configuration
# ======================================================================

DEFAULT_RADIOMICS_PARAMS = {
    "featureClass": {
        "firstorder": {},
        "glcm": {},
        "glrlm": {},    # Gray Level Run Length Matrix
        "gldm": {},     # Gray Level Dependence Matrix
    },
    "setting": {
        "binWidth": 25,
        "resampledPixelSpacing": None,
        "interpolator": "sitkBSpline",
        "normalize": False,
        "normalizeScale": 1,
        "force2D": True,
        "force2Ddimension": 0,
    },
}

# Features critical for hallucination detection
CRITICAL_FEATURES = {
    "firstorder": [
        "Mean", "Median", "StandardDeviation", "Entropy",
        "10Percentile", "90Percentile", "InterquartileRange",
        "Skewness", "Kurtosis", "Energy",
    ],
    "glcm": [
        "Contrast", "Correlation", "Homogeneity1",
        "JointEntropy", "ClusterShade", "Imc1",
    ],
}

# Acceptable deviation thresholds (%)
DEVIATION_THRESHOLDS = {
    "firstorder_Mean": 2.0,
    "firstorder_Median": 2.0,
    "firstorder_Entropy": 5.0,
    "firstorder_StandardDeviation": 15.0,  # Expected to change (denoising)
    "glcm_Contrast": 20.0,                 # Expected to decrease (denoising)
    "glcm_Correlation": 5.0,
    "glcm_Homogeneity1": 10.0,
    "default": 10.0,
}


# ======================================================================
# Hallucination Analyzer
# ======================================================================

class HallucinationAnalyzer:
    """
    Radiomic feature preservation analyzer for hallucination detection.
    
    Args:
        params: PyRadiomics parameter dict (or None for defaults).
        deviation_thresholds: Dict of feature_name → max allowable % deviation.
        hu_min: HU range minimum for denormalization.
        hu_max: HU range maximum for denormalization.
    """
    
    def __init__(
        self,
        params: Optional[Dict] = None,
        deviation_thresholds: Optional[Dict[str, float]] = None,
        hu_min: int = -1000,
        hu_max: int = 1000,
    ):
        if not RADIOMICS_AVAILABLE:
            print("⚠️  pyradiomics not installed. pip install pyradiomics")
        if not SITK_AVAILABLE:
            print("⚠️  SimpleITK not installed. pip install SimpleITK")
        
        self.params = params or DEFAULT_RADIOMICS_PARAMS
        self.thresholds = deviation_thresholds or DEVIATION_THRESHOLDS
        self.hu_min = hu_min
        self.hu_max = hu_max
    
    def _denormalize(self, image: np.ndarray) -> np.ndarray:
        """Convert [-1, 1] → HU."""
        return np.clip((image + 1.0) / 2.0 * (self.hu_max - self.hu_min) + self.hu_min, self.hu_min, self.hu_max)
    
    def _numpy_to_sitk(
        self, image: np.ndarray, is_mask: bool = False
    ) -> "sitk.Image":
        """Convert numpy image to SimpleITK Image."""
        if is_mask:
            sitk_img = sitk.GetImageFromArray(image.astype(np.uint8))
            sitk_img = sitk.Cast(sitk_img, sitk.sitkInt32)
        else:
            sitk_img = sitk.GetImageFromArray(image.astype(np.float64))
        return sitk_img
    
    def _create_body_mask(self, image_hu: np.ndarray) -> np.ndarray:
        """Create a body region mask (exclude air outside patient)."""
        mask = (image_hu > -500).astype(np.uint8)
        return mask
    
    def _extract_features(
        self, image_hu: np.ndarray, mask: np.ndarray
    ) -> Dict[str, float]:
        """
        Extract radiomic features from an image.
        
        Args:
            image_hu: Image in HU values (H, W) or (N, H, W).
            mask: Binary body mask.
        
        Returns:
            Dict mapping feature_name → value.
        """
        if not RADIOMICS_AVAILABLE or not SITK_AVAILABLE:
            return self._extract_features_numpy(image_hu, mask)
        
        # Convert to SimpleITK
        sitk_image = self._numpy_to_sitk(image_hu)
        sitk_mask = self._numpy_to_sitk(mask, is_mask=True)
        
        # Extract features
        extractor = featureextractor.RadiomicsFeatureExtractor()
        
        # Enable only critical features
        extractor.disableAllFeatures()
        for feat_class in self.params.get("featureClass", {}):
            extractor.enableFeatureClassByName(feat_class)
        
        # Set binning
        settings = self.params.get("setting", {})
        for k, v in settings.items():
            if v is not None:
                extractor.settings[k] = v
        
        try:
            result = extractor.execute(sitk_image, sitk_mask)
        except Exception as e:
            print(f"  ⚠️  Feature extraction failed: {e}")
            return self._extract_features_numpy(image_hu, mask)
        
        # Filter to only feature values (skip diagnostics)
        features = {}
        for key, value in result.items():
            if key.startswith("original_"):
                clean_key = key.replace("original_", "")
                try:
                    features[clean_key] = float(value)
                except (ValueError, TypeError):
                    pass
        
        return features
    
    def _extract_features_numpy(
        self, image_hu: np.ndarray, mask: np.ndarray
    ) -> Dict[str, float]:
        """
        Fallback: Extract basic features using NumPy.
        Used when pyradiomics/SimpleITK is not available.
        """
        masked = image_hu[mask > 0]
        
        if len(masked) == 0:
            return {}
        
        features = {
            "firstorder_Mean": float(np.mean(masked)),
            "firstorder_Median": float(np.median(masked)),
            "firstorder_StandardDeviation": float(np.std(masked)),
            "firstorder_Skewness": float(
                np.mean(((masked - np.mean(masked)) / max(np.std(masked), 1e-8)) ** 3)
            ),
            "firstorder_Kurtosis": float(
                np.mean(((masked - np.mean(masked)) / max(np.std(masked), 1e-8)) ** 4)
            ),
            "firstorder_Energy": float(np.sum(masked ** 2)),
            "firstorder_10Percentile": float(np.percentile(masked, 10)),
            "firstorder_90Percentile": float(np.percentile(masked, 90)),
            "firstorder_InterquartileRange": float(
                np.percentile(masked, 75) - np.percentile(masked, 25)
            ),
        }
        
        # Basic entropy estimate
        hist, _ = np.histogram(masked, bins=64, density=False)
        hist = hist.astype(float) / np.sum(hist)
        hist = hist[hist > 0]
        features["firstorder_Entropy"] = float(-np.sum(hist * np.log2(hist)))
        
        return features
    
    # ------------------------------------------------------------------
    # Main Analysis
    # ------------------------------------------------------------------
    
    def analyze_slice(
        self,
        predicted: np.ndarray,
        ndct: np.ndarray,
    ) -> Dict:
        """
        Analyze a single 2D slice for hallucination risk.
        
        Args:
            predicted: Predicted image (H, W) in [-1, 1].
            ndct: Ground truth NDCT (H, W) in [-1, 1].
        
        Returns:
            Dict with feature preservation analysis.
        """
        # Denormalize to HU
        pred_hu = self._denormalize(predicted)
        ndct_hu = self._denormalize(ndct)
        
        # Create body mask from NDCT
        mask = self._create_body_mask(ndct_hu)
        
        if mask.sum() < 100:
            return {"status": "skipped", "reason": "insufficient body region"}
        
        # Extract features
        feat_pred = self._extract_features(pred_hu, mask)
        feat_ndct = self._extract_features(ndct_hu, mask)
        
        # Compare
        return self._compare_features(feat_pred, feat_ndct)
    
    def analyze_volume(
        self,
        pred_volume: np.ndarray,
        ndct_volume: np.ndarray,
        sample_every: int = 10,
    ) -> Dict:
        """
        Analyze a full 3D volume for hallucination risk.
        
        Samples slices at regular intervals for efficiency.
        
        Args:
            pred_volume: Predicted volume (N, H, W) in [-1, 1].
            ndct_volume: NDCT volume (N, H, W) in [-1, 1].
            sample_every: Analyze every Nth slice.
        
        Returns:
            Aggregated hallucination report.
        """
        n_slices = pred_volume.shape[0]
        slice_indices = list(range(0, n_slices, sample_every))
        
        all_deviations = {}
        n_flagged = 0
        n_analyzed = 0
        
        for idx in slice_indices:
            result = self.analyze_slice(pred_volume[idx], ndct_volume[idx])
            
            if result.get("status") == "skipped":
                continue
            
            n_analyzed += 1
            
            if result.get("has_hallucination_risk", False):
                n_flagged += 1
            
            # Accumulate deviations
            for feat_name, dev in result.get("deviations", {}).items():
                if feat_name not in all_deviations:
                    all_deviations[feat_name] = []
                all_deviations[feat_name].append(dev)
        
        # Aggregate
        report = {
            "n_slices_analyzed": n_analyzed,
            "n_slices_flagged": n_flagged,
            "hallucination_rate": n_flagged / max(n_analyzed, 1),
            "feature_preservation": {},
        }
        
        for feat_name, devs in all_deviations.items():
            devs_arr = np.array(devs)
            threshold = self.thresholds.get(feat_name,
                                             self.thresholds.get("default", 10.0))
            
            report["feature_preservation"][feat_name] = {
                "mean_deviation_pct": float(devs_arr.mean()),
                "max_deviation_pct": float(devs_arr.max()),
                "threshold_pct": threshold,
                "preserved": bool(devs_arr.mean() < threshold),
            }
        
        # Overall verdict
        n_preserved = sum(
            1 for v in report["feature_preservation"].values()
            if v["preserved"]
        )
        n_total = len(report["feature_preservation"])
        report["preservation_rate"] = n_preserved / max(n_total, 1)
        report["verdict"] = (
            "✅ PASS" if report["preservation_rate"] >= 0.8
            else "⚠️ REVIEW" if report["preservation_rate"] >= 0.6
            else "❌ FAIL"
        )
        
        return report
    
    def _compare_features(
        self,
        feat_pred: Dict[str, float],
        feat_ndct: Dict[str, float],
    ) -> Dict:
        """Compare features and compute deviations."""
        deviations = {}
        flagged_features = []
        
        for feat_name, ndct_val in feat_ndct.items():
            if feat_name not in feat_pred:
                continue
            
            pred_val = feat_pred[feat_name]
            
            # Relative deviation (%)
            if abs(ndct_val) > 1e-8:
                dev = abs(pred_val - ndct_val) / abs(ndct_val) * 100
            else:
                dev = abs(pred_val - ndct_val) * 100
            
            deviations[feat_name] = dev
            
            threshold = self.thresholds.get(feat_name,
                                             self.thresholds.get("default", 10.0))
            
            if dev > threshold:
                flagged_features.append({
                    "feature": feat_name,
                    "ndct_value": ndct_val,
                    "pred_value": pred_val,
                    "deviation_pct": dev,
                    "threshold_pct": threshold,
                })
        
        return {
            "deviations": deviations,
            "flagged_features": flagged_features,
            "has_hallucination_risk": len(flagged_features) > 0,
        }
    
    # ------------------------------------------------------------------
    # Report Export
    # ------------------------------------------------------------------
    
    def save_report(self, report: Dict, path: str) -> None:
        """Save hallucination analysis report as JSON."""
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"  ✅ Hallucination report saved: {filepath}")
