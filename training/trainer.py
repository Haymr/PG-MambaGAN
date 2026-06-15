"""
PG-MambaGAN — PyTorch WGAN-GP Trainer (VRAM-Optimized)

Reference:
    Discriminator architecture based on Pix2Pix PatchGAN (Isola et al., 2017) 
    and Wassertein GAN-GP (Gulrajani et al., 2017).

Full training loop with:
    - BFloat16 Mixed Precision (AMP)
    - Gradient Accumulation — physical batch 1-2, effective 8-16
    - EMA (Exponential Moving Average) for generator
    - n-critic discriminator training
    - Cosine annealing + warmup learning rate schedule
    - Weights & Biases logging (losses, images, gradients, VRAM)
    - Cross-platform checkpoint save/resume

Usage:
    trainer = Trainer(config, generator, discriminator, ...)
    trainer.train(train_loader, val_loader)
"""

import os
import copy
import time
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.losses.standard_loss import (
    wasserstein_g_loss,
    wasserstein_d_loss,
    gradient_penalty,
)


class Trainer:
    """
    WGAN-GP Trainer with VRAM optimization.
    
    Args:
        config: Training configuration dict.
        generator: Generator model (VSSUNet or UNetBaseline).
        discriminator: PatchDiscriminator model.
        g_loss_fns: Dict of generator loss functions.
        device: Compute device ("cuda", "cpu").
        output_dir: Directory for checkpoints and logs.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        generator: nn.Module,
        discriminator: nn.Module,
        g_loss_fns: Dict[str, nn.Module],
        device: str = "cuda",
        output_dir: str = "experiments",
    ):
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "samples").mkdir(exist_ok=True)
        
        # Models
        self.G = generator.to(device)
        self.D = discriminator.to(device)
        self.g_loss_fns = {k: v.to(device) for k, v in g_loss_fns.items()}
        
        # Training params
        tc = config["training"]
        self.epochs = tc["epochs"]
        self.n_critic = tc.get("n_critic", 5)
        self.grad_accum = tc.get("gradient_accumulation", 8)
        self.lambda_gp = config["loss"].get("gradient_penalty", 10.0)
        
        # Loss weights (L1 and NPS use dynamic scheduling — start values)
        lc = config["loss"]
        self.loss_weights = {
            "adv": lc.get("lambda_adv", 1.0),
            "l1": lc.get("l1_start", 50.0),
            "perceptual": lc.get("lambda_perceptual", 10.0),
            "nps": lc.get("nps_start", 2.0),
            "freq": lc.get("lambda_freq", 1.0),
        }
        
        # ══════════════════════════════════════════════
        # VRAM Optimization #1: Mixed Precision (AMP)
        # ══════════════════════════════════════════════
        self.use_amp = tc.get("mixed_precision", True) and device == "cuda"
        precision = tc.get("precision_dtype", "bfloat16")
        self.amp_dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16
        self.scaler_G = torch.amp.GradScaler(enabled=self.use_amp and self.amp_dtype == torch.float16)
        self.scaler_D = torch.amp.GradScaler(enabled=self.use_amp and self.amp_dtype == torch.float16)
        
        # Optimizers
        self.opt_G = torch.optim.AdamW(
            self.G.parameters(),
            lr=tc.get("learning_rate_g", 1e-4),
            betas=(tc.get("beta1", 0.0), tc.get("beta2", 0.99)),
            weight_decay=1e-4,
        )
        self.opt_D = torch.optim.AdamW(
            self.D.parameters(),
            lr=tc.get("learning_rate_d", 1e-4),
            betas=(tc.get("beta1", 0.0), tc.get("beta2", 0.99)),
            weight_decay=1e-4,
        )
        
        # ══════════════════════════════════════════════
        # Learning Rate Scheduler (Cosine + Warmup)
        # ══════════════════════════════════════════════
        warmup = tc.get("warmup_epochs", 5)
        self.scheduler_G = self._build_scheduler(self.opt_G, warmup)
        self.scheduler_D = self._build_scheduler(self.opt_D, warmup)
        
        # ══════════════════════════════════════════════
        # EMA (Exponential Moving Average)
        # ══════════════════════════════════════════════
        self.ema_decay = tc.get("ema_decay", 0.999)
        self.G_ema = copy.deepcopy(self.G)
        self.G_ema.eval()
        for p in self.G_ema.parameters():
            p.requires_grad = False
        
        # State
        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        
        # W&B
        self.wandb_run = None
        self._init_wandb(config)
    
    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------
    
    def _build_scheduler(self, optimizer, warmup_epochs):
        """Build LR scheduler. Supports 'cosine' (default) and 'constant'."""
        from torch.optim.lr_scheduler import (
            CosineAnnealingLR, LinearLR, SequentialLR, LambdaLR
        )
        
        sched_type = self.config.get("training", {}).get("scheduler", "cosine")
        
        if sched_type == "constant":
            return LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
        
        # Default: Cosine with warmup
        warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=self.epochs - warmup_epochs)
        
        return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])
    
    # ------------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------------
    
    def _init_wandb(self, config):
        """Initialize Weights & Biases (if available)."""
        try:
            import wandb
            
            wc = config.get("wandb", {})
            if wc.get("project"):
                self.wandb_run = wandb.init(
                    project=wc["project"],
                    entity=wc.get("entity"),
                    config=config,
                    resume="allow",
                )
                print(f"  ✅ W&B initialized: {wandb.run.url}")
            else:
                print("  ℹ️  W&B project not set — logging disabled")
        except ImportError:
            print("  ℹ️  wandb not installed — logging disabled")
        except Exception as e:
            print(f"  ⚠️  W&B init failed: {e}")
    
    # ------------------------------------------------------------------
    # EMA Update
    # ------------------------------------------------------------------
    
    @torch.no_grad()
    def _update_ema(self):
        """Update EMA generator weights and synchronize buffers."""
        # 1. Update weights (EMA)
        for p_ema, p in zip(self.G_ema.parameters(), self.G.parameters()):
            p_ema.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)
            
        # 2. Synchronize buffers (Exact copy)
        for b_ema, b in zip(self.G_ema.buffers(), self.G.buffers()):
            b_ema.data.copy_(b.data)
    
    # ------------------------------------------------------------------
    # Single Discriminator Step
    # ------------------------------------------------------------------
    
    def _train_discriminator(
        self, ldct: torch.Tensor, ndct: torch.Tensor
    ) -> Dict[str, float]:
        """Train discriminator for one accumulated step."""
        self.D.train()
        d_losses = {"d_real": 0, "d_fake": 0, "d_gp": 0, "d_total": 0}
        
        with torch.amp.autocast(
            device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp
        ):
            # Generate fake
            with torch.no_grad():
                fake = self.G(ldct)
            
            # Real score
            real_input = torch.cat([ldct, ndct], dim=1)
            d_real = self.D(real_input)
            
            # Fake score
            fake_input = torch.cat([ldct, fake.detach()], dim=1)
            d_fake = self.D(fake_input)
            
            # Wasserstein loss
            d_loss = wasserstein_d_loss(d_real, d_fake)

        # Gradient penalty (computed outside AMP for stability)
        # Note: Casting to float() for mixed precision stability
        gp = gradient_penalty(
            self.D, ndct.float(), fake.detach().float(), ldct.float(), self.lambda_gp
        )

        total_d_loss = (d_loss + gp) / self.grad_accum
        
        if self.use_amp and self.amp_dtype == torch.float16:
            self.scaler_D.scale(total_d_loss).backward()
        else:
            total_d_loss.backward()

        d_losses["d_real"] = d_real.mean().item() / self.grad_accum
        d_losses["d_fake"] = d_fake.mean().item() / self.grad_accum
        d_losses["d_gp"] = gp.item() / self.grad_accum
        d_losses["d_total"] = total_d_loss.item()
        
        return d_losses
    
    # ------------------------------------------------------------------
    # Single Generator Step
    # ------------------------------------------------------------------
    
    def _train_generator(
        self, ldct: torch.Tensor, ndct: torch.Tensor
    ) -> Dict[str, float]:
        """Train generator for one accumulated step."""
        self.G.train()
        g_losses = {}
        
        # ██ CRITICAL: Freeze D — both gradients AND buffer updates ██
        # 1. requires_grad=False prevents gradient leakage into D's .grad buffers.
        # 2. D.eval() stops spectral_norm power iteration (weight_u/weight_v)
        #    from updating with fake-only data, which would corrupt the
        #    Lipschitz constraint and send poisoned gradients back to G.
        for p in self.D.parameters():
            p.requires_grad = False
        self.D.eval()
        
        with torch.amp.autocast(
            device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp
        ):
            # Generate
            fake = self.G(ldct)
            
            # Adversarial loss
            fake_input = torch.cat([ldct, fake], dim=1)
            d_fake = self.D(fake_input)
            loss_adv = wasserstein_g_loss(d_fake) * self.loss_weights["adv"]
            
            total_g_loss = loss_adv
            g_losses["g_adv"] = loss_adv.item() / self.grad_accum
            
            # L1
            if "l1" in self.g_loss_fns:
                loss_l1 = self.g_loss_fns["l1"](fake, ndct) * self.loss_weights["l1"]
                total_g_loss = total_g_loss + loss_l1
                g_losses["g_l1"] = loss_l1.item() / self.grad_accum
            
            # Perceptual
            if "perceptual" in self.g_loss_fns:
                loss_perc = self.g_loss_fns["perceptual"](fake, ndct) * self.loss_weights["perceptual"]
                total_g_loss = total_g_loss + loss_perc
                g_losses["g_perc"] = loss_perc.item() / self.grad_accum


        # Frequency loss (outside AMP — uses FFT which may not support bfloat16)
        if "freq" in self.g_loss_fns:
            loss_freq = self.g_loss_fns["freq"](fake.float(), ndct.float()) * self.loss_weights["freq"]
            total_g_loss = total_g_loss.float() + loss_freq
            g_losses["g_freq"] = loss_freq.item() / self.grad_accum

        # NPS loss (outside AMP — uses FFT which may not support bfloat16)
        if "nps" in self.g_loss_fns:
            loss_nps, tissue_losses = self.g_loss_fns["nps"](fake.float(), ndct.float())
            loss_nps = loss_nps * self.loss_weights["nps"]
            total_g_loss = total_g_loss + loss_nps
            g_losses["g_nps"] = loss_nps.item() / self.grad_accum
            # Per-tissue NPS losses → W&B (val is already float from .item() in anatomy_nps)
            if tissue_losses is not None:
                for tissue_name, val in tissue_losses.items():
                    g_losses[f"g_nps_{tissue_name}"] = val / self.grad_accum

        scaled_loss = total_g_loss / self.grad_accum
        
        if self.use_amp and self.amp_dtype == torch.float16:
            self.scaler_G.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        # ██ CRITICAL: Unfreeze D for its own training steps ██
        for p in self.D.parameters():
            p.requires_grad = True
        self.D.train()
        
        g_losses["g_total"] = scaled_loss.item()
        
        return g_losses
    
    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Run validation using EMA generator."""
        self.G_ema.eval()
        
        from evaluation.metrics import get_body_mask, compute_psnr

        total_l1 = 0.0
        total_l1_hu = 0.0
        total_psnr = 0.0
        n_samples = 0
        
        for batch in val_loader:
            ldct = batch["ldct"].to(self.device)
            ndct = batch["ndct"].to(self.device)
            
            with torch.amp.autocast(
                device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp
            ):
                fake = self.G_ema(ldct)
            
            # Convert to numpy for metrics
            fake_np = fake.cpu().float().numpy()
            ndct_np = ndct.cpu().float().numpy()
            
            batch_l1 = 0.0
            batch_l1_hu = 0.0
            batch_psnr = 0.0
            
            for i in range(ldct.shape[0]):
                # HU Denormalization for mask generation (assuming [-1, 1] -> [-1000, 1000])
                target_hu = (ndct_np[i, 0] + 1.0) / 2.0 * 2000.0 - 1000.0
                pred_hu = (fake_np[i, 0] + 1.0) / 2.0 * 2000.0 - 1000.0
                
                body_mask = get_body_mask(target_hu)

                # Masked L1
                if body_mask.sum() > 0:
                    l1_val = np.mean(np.abs(fake_np[i, 0][body_mask == 1] - ndct_np[i, 0][body_mask == 1]))
                    batch_l1 += l1_val
                    
                    l1_val_hu = np.mean(np.abs(pred_hu[body_mask == 1] - target_hu[body_mask == 1]))
                    batch_l1_hu += l1_val_hu

                # Masked PSNR
                psnr_val = compute_psnr(fake_np[i, 0], ndct_np[i, 0], data_range=2.0, body_mask=body_mask)
                batch_psnr += psnr_val

            total_l1 += batch_l1
            total_l1_hu += batch_l1_hu
            total_psnr += batch_psnr
            n_samples += ldct.shape[0]
        
        return {
            "val_l1": float(total_l1 / max(n_samples, 1)),
            "val_l1_hu": float(total_l1_hu / max(n_samples, 1)),
            "val_psnr": float(total_psnr / max(n_samples, 1)),
        }
    
    # ------------------------------------------------------------------
    # Main Training Loop
    # ------------------------------------------------------------------
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        """
        Full training loop.
        
        Args:
            train_loader: Training DataLoader.
            val_loader: Optional validation DataLoader.
        """
        log_images_every = self.config.get("wandb", {}).get("log_images_every", 10)
        
        # VRAM profiling (first batch)
        self._profile_vram()
        
        print(f"\n{'='*60}")
        print(f"  Training PG-MambaGAN")
        print(f"  Epochs: {self.start_epoch+1} → {self.epochs}")
        print(f"  Batch (physical): {train_loader.batch_size}")
        print(f"  Gradient Accumulation: {self.grad_accum}")
        print(f"  Effective Batch: {train_loader.batch_size * self.grad_accum}")
        print(f"  Mixed Precision: {self.use_amp} ({self.amp_dtype})")
        print(f"  n-critic: {self.n_critic}")
        print(f"{'='*60}\n")
        
        import math
        # Dynamic scheduling boundaries from YAML (no hardcoding)
        lc = self.config["loss"]
        l1_start = lc.get("l1_start", 50.0)
        l1_end = lc.get("l1_end", 15.0)
        nps_start = lc.get("nps_start", 2.0)
        nps_end = lc.get("nps_end", 15.0)

        for epoch in range(self.start_epoch, self.epochs):
            # Dynamic Loss Weighting (Cosine Cross-Fade)
            # L1: l1_start -> l1_end (anatomy first, then relax)
            # NPS: nps_start -> nps_end (physics gradually takes over)
            progress = epoch / max(1, self.epochs - 1)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))  # 1.0 -> 0.0

            self.loss_weights["l1"] = l1_end + (l1_start - l1_end) * cosine_factor
            self.loss_weights["nps"] = nps_end + (nps_start - nps_end) * cosine_factor

            epoch_start = time.time()
            epoch_d_losses = {}
            epoch_g_losses = {}
            n_batches = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            
            self.opt_D.zero_grad()
            self.opt_G.zero_grad()

            for step, batch in enumerate(pbar):
                ldct = batch["ldct"].to(self.device, non_blocking=True)
                ndct = batch["ndct"].to(self.device, non_blocking=True)
                
                # ⚠️ CRITICAL VRAM FIX: If gradient_checkpointing is True, the input tensor 
                # MUST require grad, otherwise checkpointing silently disables backprop for early layers.
                ldct.requires_grad_(True)
                
                # ---- Discriminator ----
                d_losses = self._train_discriminator(ldct, ndct)
                for k, v in d_losses.items():
                    epoch_d_losses[k] = epoch_d_losses.get(k, 0) + v
                
                if (step + 1) % self.grad_accum == 0 or (step + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(self.D.parameters(), max_norm=1.0)
                    if self.use_amp and self.amp_dtype == torch.float16:
                        self.scaler_D.step(self.opt_D)
                        self.scaler_D.update()
                    else:
                        self.opt_D.step()
                    self.opt_D.zero_grad()

                # ---- Generator (every n_critic steps) ----
                if (step + 1) % self.n_critic == 0:
                    g_losses = self._train_generator(ldct, ndct)
                    for k, v in g_losses.items():
                        epoch_g_losses[k] = epoch_g_losses.get(k, 0) + v

                    if ((step + 1) // self.n_critic) % self.grad_accum == 0 or (step + 1) == len(train_loader):
                        torch.nn.utils.clip_grad_norm_(self.G.parameters(), max_norm=1.0)
                        if self.use_amp and self.amp_dtype == torch.float16:
                            self.scaler_G.step(self.opt_G)
                            self.scaler_G.update()
                        else:
                            self.opt_G.step()
                        self.opt_G.zero_grad()

                        # Update EMA
                        self._update_ema()
                
                self.global_step += 1
                n_batches += 1
                
                # Progress bar
                pbar.set_postfix({
                    "D": f"{d_losses.get('d_total', 0):.3f}",
                    "G": f"{g_losses.get('g_total', 0):.3f}" if 'g_losses' in dir() else "—",
                })
            
            # ---- Epoch End ----
            epoch_time = time.time() - epoch_start
            
            # Average losses
            for k in epoch_d_losses:
                epoch_d_losses[k] /= max(n_batches, 1)
            n_g_steps = max(n_batches // self.n_critic, 1)
            for k in epoch_g_losses:
                epoch_g_losses[k] /= n_g_steps
            
            # Schedulers
            self.scheduler_G.step()
            self.scheduler_D.step()
            
            # Validation
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
            
            # Logging
            self._log_epoch(epoch, epoch_d_losses, epoch_g_losses,
                          val_metrics, epoch_time, log_images_every,
                          train_loader)
            
            # Checkpointing
            self._save_checkpoint(epoch, val_metrics)
        
        print(f"\n✅ Training complete! Best val_l1_hu: {self.best_val_loss:.1f} HU")
        
        if self.wandb_run:
            import wandb
            wandb.finish()
    
    # ------------------------------------------------------------------
    # VRAM Profiling
    # ------------------------------------------------------------------
    
    def _profile_vram(self):
        """Profile VRAM usage on first batch."""
        if not torch.cuda.is_available():
            return
        
        torch.cuda.reset_peak_memory_stats()
        
        print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  VRAM Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    
    def _log_epoch(self, epoch, d_losses, g_losses, val_metrics,
                   epoch_time, log_images_every, train_loader):
        """Log metrics to console and W&B."""
        lr = self.opt_G.param_groups[0]["lr"]
        
        print(f"\n  Epoch {epoch+1}/{self.epochs} ({epoch_time:.0f}s) "
              f"| LR: {lr:.2e}")
        print(f"  D: total={d_losses.get('d_total',0):.4f} "
              f"real={d_losses.get('d_real',0):.4f} "
              f"fake={d_losses.get('d_fake',0):.4f} "
              f"gp={d_losses.get('d_gp',0):.4f}")
        print(f"  G: total={g_losses.get('g_total',0):.4f} "
              f"adv={g_losses.get('g_adv',0):.4f} "
              f"l1={g_losses.get('g_l1',0):.4f} "
              f"nps={g_losses.get('g_nps',0):.6f}")
        
        if val_metrics:
            print(f"  Val: L1={val_metrics.get('val_l1',0):.4f} "
                  f"L1_HU={val_metrics.get('val_l1_hu',0):.1f} HU "
                  f"PSNR={val_metrics.get('val_psnr',0):.2f} dB")
        
        # VRAM
        if torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / 1e9
            print(f"  VRAM Peak: {peak_vram:.2f} GB")
        
        # W&B
        if self.wandb_run:
            import wandb
            
            log_dict = {
                **{f"d/{k}": v for k, v in d_losses.items()},
                **{f"g/{k}": v for k, v in g_losses.items()},
                **{f"val/{k}": v for k, v in val_metrics.items()},
                **{f"w/{k}": v for k, v in self.loss_weights.items()},
                "lr": lr,
                "epoch": epoch + 1,
            }
            
            if torch.cuda.is_available():
                log_dict["vram_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
            
            # Sample images
            if (epoch + 1) % log_images_every == 0:
                self._log_sample_images(train_loader, epoch)
            
            wandb.log(log_dict, step=self.global_step)
    
    def _log_sample_images(self, loader, epoch):
        """Log sample images to W&B using fixed content-rich slices.
        
        Uses G_raw (active model) for instant visual feedback.
        Validation metrics and checkpointing still use G_ema.
        """
        try:
            import wandb
            
            # Cache a fixed batch on first call so every epoch compares the SAME samples.
            # Scout multiple batches and pick the 3 slices with highest anatomical content
            # (highest pixel std → more structure + more visible noise).
            if not hasattr(self, "_fixed_sample_batch") or self._fixed_sample_batch is None:
                candidates = []  # list of (score, ldct_slice, ndct_slice)
                scout_batches = 8
                for bi, batch in enumerate(loader):
                    for i in range(batch["ldct"].shape[0]):
                        score = batch["ldct"][i, 0].std().item()
                        candidates.append((
                            score,
                            batch["ldct"][i:i+1].clone(),
                            batch["ndct"][i:i+1].clone(),
                        ))
                    if bi + 1 >= scout_batches:
                        break
                candidates.sort(key=lambda x: -x[0])
                top3 = candidates[:3]
                self._fixed_sample_batch = {
                    "ldct": torch.cat([c[1] for c in top3], dim=0),
                    "ndct": torch.cat([c[2] for c in top3], dim=0),
                }
                print(f"  [sample_log] locked 3 content-rich slices, "
                      f"std scores: {[f'{c[0]:.3f}' for c in top3]}")
            
            ldct = self._fixed_sample_batch["ldct"].to(self.device)
            ndct = self._fixed_sample_batch["ndct"].to(self.device)
            
            # Use G_raw (active model) for instant visual feedback.
            # EMA (decay=0.999) smooths out training noise which is great for
            # metrics/checkpoints, but for W&B we want to see what G learned NOW.
            with torch.no_grad():
                self.G.eval()
                fake = self.G(ldct)
                self.G.train()
            
            # Denormalize [-1,1] → [0,1] for visualization
            images = []
            for i in range(min(3, ldct.shape[0])):
                trio = torch.cat([
                    (ldct[i, 0] + 1) / 2,
                    (fake[i, 0] + 1) / 2,
                    (ndct[i, 0] + 1) / 2,
                ], dim=1)  # Side by side: LDCT | Denoised | NDCT
                images.append(wandb.Image(
                    trio.cpu().float().numpy(),
                    caption=f"LDCT | Denoised | NDCT"
                ))
            
            wandb.log({"samples": images}, step=self.global_step)
        except Exception:
            pass
    
    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    
    def _save_checkpoint(self, epoch, val_metrics):
        """Save checkpoint (periodic + best)."""
        ckpt = {
            "epoch": epoch + 1,
            "global_step": self.global_step,
            "generator": self.G.state_dict(),
            "discriminator": self.D.state_dict(),
            "g_ema": self.G_ema.state_dict(),
            "opt_G": self.opt_G.state_dict(),
            "opt_D": self.opt_D.state_dict(),
            "sched_G": self.scheduler_G.state_dict(),
            "sched_D": self.scheduler_D.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        
        ckpt_dir = self.output_dir / "checkpoints"
        
        # Latest
        torch.save(ckpt, ckpt_dir / "latest.pth")
        
        # Periodic (every 10 epochs)
        if (epoch + 1) % 10 == 0:
            torch.save(ckpt, ckpt_dir / f"epoch_{epoch+1:04d}.pth")
        
        # Best
        val_l1_hu = val_metrics.get("val_l1_hu", float("inf"))
        if val_l1_hu < self.best_val_loss:
            self.best_val_loss = val_l1_hu
            torch.save(ckpt, ckpt_dir / "best.pth")
            print(f"  ★ New best model saved (val_l1_hu={val_l1_hu:.1f} HU)")
    
    def load_checkpoint(self, path: str):
        """Resume training from a checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        
        self.G.load_state_dict(ckpt["generator"])
        self.D.load_state_dict(ckpt["discriminator"])
        self.G_ema.load_state_dict(ckpt["g_ema"])
        self.opt_G.load_state_dict(ckpt["opt_G"])
        self.opt_D.load_state_dict(ckpt["opt_D"])
        self.scheduler_G.load_state_dict(ckpt["sched_G"])
        self.scheduler_D.load_state_dict(ckpt["sched_D"])
        self.start_epoch = ckpt["epoch"]
        self.global_step = ckpt["global_step"]
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        
        print(f"  ✅ Resumed from epoch {self.start_epoch} "
              f"(step {self.global_step})")
    
    def load_finetune(self, path: str):
        """Load only G/D/EMA weights for fine-tuning (fresh optimizer & scheduler)."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        
        self.G.load_state_dict(ckpt["generator"])
        self.D.load_state_dict(ckpt["discriminator"])
        
        if "g_ema" in ckpt:
            self.G_ema.load_state_dict(ckpt["g_ema"])
        
        # Optimizer ve scheduler yüklenmez — sıfırdan başlar
        # start_epoch sıfırda kalır — 0'dan epochs'a kadar yeni eğitim
        src_epoch = ckpt.get("epoch", "?")
        print(f"  🔧 Fine-tune mode: loaded G/D/EMA weights from epoch {src_epoch}")
        print(f"     Optimizer & scheduler: fresh (not restored)")
