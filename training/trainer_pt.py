"""
PG-MambaGAN — PyTorch WGAN-GP Trainer (VRAM-Optimized)

Full training loop with:
    - BFloat16 Mixed Precision (AMP) — Revision #2
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
from pathlib import Path
from typing import Dict, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.losses.standard_pt import (
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
        
        # Loss weights
        lc = config["loss"]
        self.loss_weights = {
            "adv": lc.get("lambda_adv", 1.0),
            "l1": lc.get("lambda_l1", 100.0),
            "perceptual": lc.get("lambda_perceptual", 10.0),
            "nps": lc.get("lambda_nps", 5.0),
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
        """Cosine annealing with linear warmup."""
        from torch.optim.lr_scheduler import (
            CosineAnnealingLR, LinearLR, SequentialLR
        )
        
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
        """Update EMA generator weights."""
        for p_ema, p in zip(self.G_ema.parameters(), self.G.parameters()):
            p_ema.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)
    
    # ------------------------------------------------------------------
    # Single Discriminator Step
    # ------------------------------------------------------------------
    
    def _train_discriminator(
        self, ldct: torch.Tensor, ndct: torch.Tensor
    ) -> Dict[str, float]:
        """Train discriminator for one accumulated step."""
        self.D.train()
        d_losses = {"d_real": 0, "d_fake": 0, "d_gp": 0, "d_total": 0}
        
        self.opt_D.zero_grad()
        
        for accum_step in range(self.grad_accum):
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
            gp = gradient_penalty(
                self.D, ndct, fake.detach(), ldct, self.lambda_gp
            )
            
            total_d_loss = (d_loss + gp) / self.grad_accum
            
            if self.use_amp and self.amp_dtype == torch.float16:
                self.scaler_D.scale(total_d_loss).backward()
            else:
                total_d_loss.backward()
            
            d_losses["d_real"] += d_real.mean().item() / self.grad_accum
            d_losses["d_fake"] += d_fake.mean().item() / self.grad_accum
            d_losses["d_gp"] += gp.item() / self.grad_accum
            d_losses["d_total"] += total_d_loss.item()
        
        if self.use_amp and self.amp_dtype == torch.float16:
            self.scaler_D.step(self.opt_D)
            self.scaler_D.update()
        else:
            self.opt_D.step()
        
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
        
        self.opt_G.zero_grad()
        
        for accum_step in range(self.grad_accum):
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
                g_losses["g_adv"] = g_losses.get("g_adv", 0) + loss_adv.item() / self.grad_accum
                
                # L1
                if "l1" in self.g_loss_fns:
                    loss_l1 = self.g_loss_fns["l1"](fake, ndct) * self.loss_weights["l1"]
                    total_g_loss = total_g_loss + loss_l1
                    g_losses["g_l1"] = g_losses.get("g_l1", 0) + loss_l1.item() / self.grad_accum
                
                # Perceptual
                if "perceptual" in self.g_loss_fns:
                    loss_perc = self.g_loss_fns["perceptual"](fake, ndct) * self.loss_weights["perceptual"]
                    total_g_loss = total_g_loss + loss_perc
                    g_losses["g_perc"] = g_losses.get("g_perc", 0) + loss_perc.item() / self.grad_accum
                
            # Frequency (outside AMP — uses FFT which may be unstable)
            if "freq" in self.g_loss_fns:
                loss_freq = self.g_loss_fns["freq"](fake.float(), ndct.float()) * self.loss_weights["freq"]
                total_g_loss = total_g_loss + loss_freq
                g_losses["g_freq"] = g_losses.get("g_freq", 0) + loss_freq.item() / self.grad_accum
            
            # NPS loss (outside AMP — uses FFT which may not support bfloat16)
            if "nps" in self.g_loss_fns:
                loss_nps, tissue_losses = self.g_loss_fns["nps"](fake.float(), ndct.float())
                loss_nps = loss_nps * self.loss_weights["nps"]
                total_g_loss = total_g_loss + loss_nps
                g_losses["g_nps"] = g_losses.get("g_nps", 0) + loss_nps.item() / self.grad_accum
            
            scaled_loss = total_g_loss / self.grad_accum
            
            if self.use_amp and self.amp_dtype == torch.float16:
                self.scaler_G.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            
            g_losses["g_total"] = g_losses.get("g_total", 0) + scaled_loss.item()
        
        if self.use_amp and self.amp_dtype == torch.float16:
            self.scaler_G.step(self.opt_G)
            self.scaler_G.update()
        else:
            self.opt_G.step()
        
        # Update EMA
        self._update_ema()
        
        return g_losses
    
    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Run validation using EMA generator."""
        self.G_ema.eval()
        
        total_l1 = 0.0
        total_psnr = 0.0
        n_samples = 0
        
        for batch in val_loader:
            ldct = batch["ldct"].to(self.device)
            ndct = batch["ndct"].to(self.device)
            
            with torch.amp.autocast(
                device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp
            ):
                fake = self.G_ema(ldct)
            
            # L1
            total_l1 += nn.functional.l1_loss(fake, ndct).item() * ldct.shape[0]
            
            # PSNR (on [-1,1] range)
            mse = nn.functional.mse_loss(fake, ndct).item()
            if mse > 0:
                psnr = 10 * torch.log10(torch.tensor(4.0 / mse)).item()  # range=2 for [-1,1]
                total_psnr += psnr * ldct.shape[0]
            
            n_samples += ldct.shape[0]
        
        return {
            "val_l1": total_l1 / max(n_samples, 1),
            "val_psnr": total_psnr / max(n_samples, 1),
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
        
        for epoch in range(self.start_epoch, self.epochs):
            epoch_start = time.time()
            epoch_d_losses = {}
            epoch_g_losses = {}
            n_batches = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            
            for step, batch in enumerate(pbar):
                ldct = batch["ldct"].to(self.device, non_blocking=True)
                ndct = batch["ndct"].to(self.device, non_blocking=True)
                
                # ---- Discriminator ----
                d_losses = self._train_discriminator(ldct, ndct)
                for k, v in d_losses.items():
                    epoch_d_losses[k] = epoch_d_losses.get(k, 0) + v
                
                # ---- Generator (every n_critic steps) ----
                if (step + 1) % self.n_critic == 0:
                    g_losses = self._train_generator(ldct, ndct)
                    for k, v in g_losses.items():
                        epoch_g_losses[k] = epoch_g_losses.get(k, 0) + v
                
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
        
        print(f"\n✅ Training complete! Best val_l1: {self.best_val_loss:.6f}")
        
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
        print(f"  VRAM Total: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
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
        """Log sample images to W&B."""
        try:
            import wandb
            
            batch = next(iter(loader))
            ldct = batch["ldct"][:4].to(self.device)
            ndct = batch["ndct"][:4].to(self.device)
            
            with torch.no_grad():
                fake = self.G_ema(ldct)
            
            # Denormalize [-1,1] → [0,1] for visualization
            images = []
            for i in range(min(4, ldct.shape[0])):
                trio = torch.cat([
                    (ldct[i, 0] + 1) / 2,
                    (fake[i, 0] + 1) / 2,
                    (ndct[i, 0] + 1) / 2,
                ], dim=1)  # Side by side: LDCT | Denoised | NDCT
                images.append(wandb.Image(
                    trio.cpu().numpy(),
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
        val_l1 = val_metrics.get("val_l1", float("inf"))
        if val_l1 < self.best_val_loss:
            self.best_val_loss = val_l1
            torch.save(ckpt, ckpt_dir / "best.pth")
            print(f"  ★ New best model saved (val_l1={val_l1:.6f})")
    
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
