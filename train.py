"""
PG-MambaGAN — Training Entry Point

Usage:
    # Windows/Linux:
    python train.py --config configs/default.yaml --data-path D:\\data\\mayo

    # Colab:
    python train.py --config configs/default.yaml \
                    --data-path /content/drive/MyDrive/data/mayo

    # Resume:
    python train.py --config configs/default.yaml \
                    --resume experiments/checkpoints/latest.pth
"""

import argparse
import sys
from pathlib import Path

import yaml
import torch

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from setup.environment import Environment
from data.patient_manifest import PatientManifest
from data.dataset import LDCTDataset
from models.generators.vss_unet import VSSUNet
from models.generators.unet_baseline import UNetBaseline
from models.discriminators.patch_disc import PatchDiscriminator
from models.losses.standard_loss import L1Loss
from models.losses.perceptual_loss import PerceptualLoss
from models.losses.frequency_loss import FrequencyLoss
from models.losses.anatomy_nps import AnatomyAwareNPSLoss
from training.trainer import Trainer


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def merge_cli_args(config: dict, args) -> dict:
    """Override config values with CLI arguments."""
    if args.data_path:
        config.setdefault("paths", {})
        config["paths"]["data_root"] = args.data_path
    
    if args.output_dir:
        config.setdefault("paths", {})
        config["paths"]["output_dir"] = args.output_dir
    
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    
    if args.accumulation:
        config["training"]["gradient_accumulation"] = args.accumulation
    
    return config


def build_generator(config: dict) -> torch.nn.Module:
    """Build generator model from config."""
    gen_type = config["model"].get("generator", "vss_unet")
    gc = config.get("generator", {})
    
    if gen_type == "vss_unet":
        model = VSSUNet(
            in_channels=config["model"].get("channels", 1),
            out_channels=config["model"].get("channels", 1),
            embed_dim=gc.get("embed_dim", 96),
            depths=gc.get("depths", [2, 2, 4, 2]),
            patch_size=gc.get("patch_size", 4),
            drop_rate=gc.get("drop_rate", 0.0),
            drop_path_rate=gc.get("drop_path_rate", 0.2),
            use_checkpoint=gc.get("gradient_checkpointing", True),
        )
    elif gen_type == "unet_baseline":
        model = UNetBaseline(
            in_channels=config["model"].get("channels", 1),
            out_channels=config["model"].get("channels", 1),
        )
    else:
        raise ValueError(f"Unknown generator: {gen_type}")
    
    params = model.get_param_count()
    print(f"  Generator: {gen_type} ({params['total_M']} params)")
    return model


def build_discriminator(config: dict) -> torch.nn.Module:
    """Build discriminator from config."""
    dc = config.get("discriminator", {})
    
    model = PatchDiscriminator(
        in_channels=config["model"].get("channels", 1) * 2,
        filters=dc.get("filters", [64, 128, 256, 512]),
        use_spectral_norm=dc.get("use_spectral_norm", True),
    )
    
    params = model.get_param_count()
    print(f"  Discriminator: PatchGAN-SN ({params['total_M']} params)")
    return model


def build_loss_fns(config: dict) -> dict:
    """Build generator loss functions from config."""
    lc = config.get("loss", {})
    losses = {}
    
    # L1 (always present)
    losses["l1"] = L1Loss()
    
    # Perceptual
    if lc.get("lambda_perceptual", 0) > 0:
        losses["perceptual"] = PerceptualLoss(use_lpips=False)
    
    # Frequency
    if lc.get("lambda_freq", 0) > 0:
        losses["freq"] = FrequencyLoss()
    
    # Anatomy-Aware NPS
    if lc.get("lambda_nps", 0) > 0 or lc.get("nps_start", 0) > 0:
        losses["nps"] = AnatomyAwareNPSLoss(
            tissue_weights=lc.get("nps_tissue_weights"),
            n_patches=8,
            close_kernel=lc.get("nps_morphological_kernel", 5),
        )
    
    print(f"  Losses: {list(losses.keys())}")
    return losses


def main():
    parser = argparse.ArgumentParser(description="PG-MambaGAN Training")
    
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Override data root path")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint for resuming")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override physical batch size")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--accumulation", type=int, default=None,
                        help="Override gradient accumulation steps")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Path to patient manifest JSON")
    
    args = parser.parse_args()
    
    # ---- Environment ----
    env = Environment()
    print(env.summary())
    
    # ---- Config ----
    config = load_config(args.config)
    config = merge_cli_args(config, args)
    
    # VRAM-aware defaults
    if args.batch_size is None and args.accumulation is None:
        rec = env.recommend_training_config()
        if config["training"].get("batch_size") is None:
            config["training"]["batch_size"] = rec["batch_size"]
            config["training"]["gradient_accumulation"] = rec["gradient_accumulation"]
            print(f"  ℹ️  Auto VRAM config: batch={rec['batch_size']}, "
                  f"accum={rec['gradient_accumulation']}")
    
    device = env.device
    
    # ---- Data ----
    data_root = config.get("paths", {}).get("data_root") or str(env.resolve_data_path())
    manifest_path = args.manifest or str(Path(data_root) / "patient_manifest.json")
    
    print(f"\n  Loading manifest: {manifest_path}")
    manifest = PatientManifest.load(manifest_path)
    manifest.verify_no_leakage()
    
    tc = config["training"]
    
    train_dataset = LDCTDataset(
        manifest=manifest, split="train",
        data_root=data_root,
        img_size=config["model"].get("img_size", 512),
        augment=True,
    )
    train_loader = train_dataset.get_dataloader(
        batch_size=tc["batch_size"],
        num_workers=4 if env.runtime != Environment.COLAB else 2,
    )
    
    val_dataset = LDCTDataset(
        manifest=manifest, split="val",
        data_root=data_root,
        img_size=config["model"].get("img_size", 512),
        augment=False,
    )
    val_loader = val_dataset.get_dataloader(
        batch_size=tc["batch_size"],
        num_workers=2,
    )
    
    # ---- Models ----
    print("\n  Building models...")
    generator = build_generator(config)
    discriminator = build_discriminator(config)
    loss_fns = build_loss_fns(config)
    
    # ---- Trainer ----
    output_dir = config.get("paths", {}).get("output_dir", "experiments")
    trainer = Trainer(
        config=config,
        generator=generator,
        discriminator=discriminator,
        g_loss_fns=loss_fns,
        device=device,
        output_dir=output_dir,
    )
    
    # Resume
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # ---- Train ----
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
