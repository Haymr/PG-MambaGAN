"""
PG-MambaGAN — Google Colab Setup Script

Run this script to set up the full training environment on Colab:
    !python setup/colab_setup.py

Steps:
    1. Install all Python dependencies
    2. Install mamba-ssm with CUDA support
    3. Verify GPU and VRAM
    4. Set up Weights & Biases
    5. Print environment summary
"""

import subprocess
import sys
import os


def run_cmd(cmd: str, desc: str = "") -> None:
    """Run a shell command with error handling."""
    if desc:
        print(f"\n{'='*50}")
        print(f"  {desc}")
        print(f"{'='*50}")
    
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True
    )
    
    if result.returncode != 0:
        print(f"  ❌ Error: {result.stderr[:500]}")
    else:
        # Show only last few lines of output
        output_lines = result.stdout.strip().split("\n")
        for line in output_lines[-5:]:
            print(f"  {line}")
        print("  ✅ Done")


def main():
    print("=" * 60)
    print("  PG-MambaGAN — Colab Environment Setup")
    print("=" * 60)
    
    # -----------------------------------------------------------
    # 1. Install core dependencies
    # -----------------------------------------------------------
    run_cmd(
        f"{sys.executable} -m pip install -q "
        "torch torchvision torchmetrics "
        "monai SimpleITK pydicom nibabel "
        "wandb lpips pytorch-fid "
        "scikit-image scipy matplotlib "
        "pyyaml tqdm pandas opencv-python Pillow "
        "pyradiomics scikit-learn",
        "Installing core dependencies"
    )
    
    # -----------------------------------------------------------
    # 2. Install mamba-ssm (CUDA required)
    # -----------------------------------------------------------
    run_cmd(
        f"{sys.executable} -m pip install -q causal-conv1d mamba-ssm",
        "Installing mamba-ssm (CUDA-accelerated SSM)"
    )
    
    # -----------------------------------------------------------
    # 3. GPU Verification
    # -----------------------------------------------------------
    print(f"\n{'='*50}")
    print("  GPU Verification")
    print(f"{'='*50}")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            print(f"  ✅ GPU: {gpu_name}")
            print(f"  ✅ VRAM: {vram:.1f} GB")
            print(f"  ✅ CUDA: {torch.version.cuda}")
            
            # VRAM recommendations
            if vram < 12:
                print(f"  ⚠️  Low VRAM ({vram:.0f}GB). Use batch_size=1, accumulation=8")
            elif vram < 20:
                print(f"  ℹ️  Standard VRAM ({vram:.0f}GB). Use batch_size=1, accumulation=8")
            else:
                print(f"  ✅ High VRAM ({vram:.0f}GB). Use batch_size=2-4, accumulation=2-4")
        else:
            print("  ❌ No GPU! Go to Runtime > Change runtime type > GPU")
            return
    except ImportError:
        print("  ❌ PyTorch not installed properly!")
        return
    
    # -----------------------------------------------------------
    # 4. Mamba SSM Verification
    # -----------------------------------------------------------
    print(f"\n{'='*50}")
    print("  Mamba SSM Verification")
    print(f"{'='*50}")
    
    try:
        import mamba_ssm
        print(f"  ✅ mamba-ssm version: {mamba_ssm.__version__}")
    except ImportError:
        print("  ❌ mamba-ssm failed to install.")
        print("  Try: pip install mamba-ssm --no-build-isolation")
    except Exception as e:
        print(f"  ⚠️  mamba-ssm imported with warning: {e}")
    
    # -----------------------------------------------------------
    # 5. Google Drive Mount (interactive)
    # -----------------------------------------------------------
    print(f"\n{'='*50}")
    print("  Google Drive Mount")
    print(f"{'='*50}")
    
    try:
        from google.colab import drive
        if not os.path.exists("/content/drive/MyDrive"):
            drive.mount("/content/drive")
            print("  ✅ Google Drive mounted at /content/drive")
        else:
            print("  ✅ Google Drive already mounted")
    except ImportError:
        print("  ℹ️  Not running on Colab — skipping Drive mount")
    
    # -----------------------------------------------------------
    # 6. W&B Setup
    # -----------------------------------------------------------
    print(f"\n{'='*50}")
    print("  Weights & Biases Setup")
    print(f"{'='*50}")
    
    try:
        import wandb
        
        # Check if already logged in
        if wandb.api.api_key:
            print(f"  ✅ W&B authenticated (key: ...{wandb.api.api_key[-6:]})")
        else:
            print("  ℹ️  Run `wandb login` or set WANDB_API_KEY to enable tracking")
    except Exception:
        print("  ℹ️  W&B not configured. Run `wandb login` later.")
    
    # -----------------------------------------------------------
    # 7. Final Summary
    # -----------------------------------------------------------
    print(f"\n{'='*60}")
    print("  ✅ PG-MambaGAN SETUP COMPLETE")
    print(f"{'='*60}")
    print("  Next steps:")
    print("    1. Set your data path: --data-path /content/drive/MyDrive/data/mayo")
    print("    2. Start training: !python train.py --config configs/default.yaml")
    print("    3. Monitor: https://wandb.ai")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
