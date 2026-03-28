"""
PG-MambaGAN Environment Detection & Configuration

Automatically detects the runtime environment (Colab, Windows, Linux)
and configures paths, GPU, and dependencies accordingly.

Usage:
    from setup.environment import Environment
    env = Environment()
    print(env.summary())
"""

import os
import sys
import platform
from pathlib import Path
from typing import Optional, Dict, Any


class Environment:
    """
    Detects and configures the runtime environment for PG-MambaGAN.
    
    Supports:
        - Google Colab (Linux + /content/ directory)
        - Windows (local GPU workstation)
        - Linux (local GPU workstation or cloud VM)
    """
    
    # Runtime type constants
    COLAB = "colab"
    WINDOWS = "windows"
    LINUX = "linux"
    
    def __init__(self):
        self.runtime = self._detect_runtime()
        self.device = self._detect_device()
        self.gpu_info = self._get_gpu_info()
        self.mamba_available = self._check_mamba()
        self.python_version = platform.python_version()
        self.project_root = self._find_project_root()
    
    # ------------------------------------------------------------------
    # Runtime Detection
    # ------------------------------------------------------------------
    
    def _detect_runtime(self) -> str:
        """Detect whether we're running on Colab, Windows, or Linux."""
        # Colab detection: check for the google.colab module
        try:
            import google.colab  # noqa: F401
            return self.COLAB
        except ImportError:
            pass
        
        # OS-based detection
        system = platform.system().lower()
        if system == "windows":
            return self.WINDOWS
        else:
            return self.LINUX
    
    # ------------------------------------------------------------------
    # GPU / CUDA Detection
    # ------------------------------------------------------------------
    
    def _detect_device(self) -> str:
        """Detect the best available compute device."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"  # Apple Silicon — limited, mainly for testing
            else:
                return "cpu"
        except ImportError:
            return "cpu"
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get detailed GPU information."""
        info = {
            "available": False,
            "name": None,
            "vram_gb": None,
            "cuda_version": None,
            "count": 0,
        }
        
        try:
            import torch
            if torch.cuda.is_available():
                info["available"] = True
                info["count"] = torch.cuda.device_count()
                info["name"] = torch.cuda.get_device_name(0)
                info["cuda_version"] = torch.version.cuda
                
                # VRAM in GB
                total_mem = torch.cuda.get_device_properties(0).total_mem
                info["vram_gb"] = round(total_mem / (1024 ** 3), 1)
        except ImportError:
            pass
        
        return info
    
    # ------------------------------------------------------------------
    # Mamba SSM Check
    # ------------------------------------------------------------------
    
    def _check_mamba(self) -> bool:
        """Check if mamba-ssm is installed and functional."""
        try:
            import mamba_ssm  # noqa: F401
            return True
        except ImportError:
            return False
    
    # ------------------------------------------------------------------
    # Path Utilities
    # ------------------------------------------------------------------
    
    def _find_project_root(self) -> Path:
        """Find the PG-MambaGAN project root directory."""
        # Try finding from current file location
        current = Path(__file__).resolve().parent.parent
        if (current / "train.py").exists():
            return current
        
        # Try CWD
        cwd = Path.cwd()
        if (cwd / "train.py").exists():
            return cwd
        
        # Fallback
        return cwd
    
    def resolve_data_path(self, data_root: Optional[str] = None) -> Path:
        """
        Resolve the data directory path based on environment.
        
        Priority:
            1. Explicit data_root argument
            2. DATA_ROOT environment variable
            3. Default path for the runtime environment
        
        Args:
            data_root: Optional explicit path to data directory.
            
        Returns:
            Resolved Path object (platform-normalized).
        """
        if data_root:
            return Path(data_root).resolve()
        
        env_root = os.environ.get("DATA_ROOT")
        if env_root:
            return Path(env_root).resolve()
        
        # Environment-specific defaults
        defaults = {
            self.COLAB: "/content/drive/MyDrive/data",
            self.WINDOWS: str(self.project_root / "data"),
            self.LINUX: str(self.project_root / "data"),
        }
        
        return Path(defaults[self.runtime]).resolve()
    
    def resolve_output_path(self, output_dir: Optional[str] = None) -> Path:
        """Resolve the output/experiments directory."""
        if output_dir:
            return Path(output_dir).resolve()
        
        return (self.project_root / "experiments").resolve()
    
    # ------------------------------------------------------------------
    # VRAM-Aware Configuration
    # ------------------------------------------------------------------
    
    def recommend_training_config(self) -> Dict[str, Any]:
        """
        Recommend training configuration based on available VRAM.
        
        Returns batch_size, gradient_accumulation, and mixed_precision 
        settings optimized for the detected GPU.
        """
        vram = self.gpu_info.get("vram_gb", 0) or 0
        
        if vram >= 40:
            # A100 / A6000
            return {
                "batch_size": 4,
                "gradient_accumulation": 2,
                "mixed_precision": True,
                "gradient_checkpointing": True,
                "note": f"A100-class GPU detected ({vram}GB). Comfortable settings.",
            }
        elif vram >= 20:
            # RTX 4090 / RTX 3090
            return {
                "batch_size": 2,
                "gradient_accumulation": 4,
                "mixed_precision": True,
                "gradient_checkpointing": True,
                "note": f"High-end GPU detected ({vram}GB). Moderate batch with accumulation.",
            }
        elif vram >= 10:
            # RTX 3080 / T4
            return {
                "batch_size": 1,
                "gradient_accumulation": 8,
                "mixed_precision": True,
                "gradient_checkpointing": True,
                "note": f"Mid-range GPU detected ({vram}GB). Minimal batch, heavy accumulation.",
            }
        else:
            # Low VRAM or CPU
            return {
                "batch_size": 1,
                "gradient_accumulation": 8,
                "mixed_precision": False,
                "gradient_checkpointing": True,
                "note": f"Low VRAM or CPU ({vram}GB). Training will be very slow.",
            }
    
    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    
    def summary(self) -> str:
        """Return a formatted summary of the environment."""
        rec = self.recommend_training_config()
        
        lines = [
            "=" * 60,
            "  PG-MambaGAN Environment Summary",
            "=" * 60,
            f"  Runtime:        {self.runtime.upper()}",
            f"  Python:         {self.python_version}",
            f"  Device:         {self.device}",
            f"  GPU:            {self.gpu_info['name'] or 'N/A'}",
            f"  VRAM:           {self.gpu_info['vram_gb'] or 'N/A'} GB",
            f"  CUDA:           {self.gpu_info['cuda_version'] or 'N/A'}",
            f"  GPU Count:      {self.gpu_info['count']}",
            f"  Mamba SSM:      {'✅ Available' if self.mamba_available else '❌ Not installed'}",
            f"  Project Root:   {self.project_root}",
            "-" * 60,
            "  Recommended Training Config:",
            f"    batch_size:             {rec['batch_size']}",
            f"    gradient_accumulation:  {rec['gradient_accumulation']}",
            f"    mixed_precision:        {rec['mixed_precision']}",
            f"    gradient_checkpointing: {rec['gradient_checkpointing']}",
            f"    note: {rec['note']}",
            "=" * 60,
        ]
        
        return "\n".join(lines)


# ------------------------------------------------------------------
# Convenience: run as script to print environment info
# ------------------------------------------------------------------

if __name__ == "__main__":
    env = Environment()
    print(env.summary())
    
    if not env.gpu_info["available"]:
        print("\n⚠️  No CUDA GPU detected. Training requires a CUDA-capable GPU.")
        print("    For Colab: Runtime > Change runtime type > GPU")
    
    if not env.mamba_available:
        print("\n⚠️  mamba-ssm not installed. Install with:")
        print("    pip install mamba-ssm causal-conv1d")
