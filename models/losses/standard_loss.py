"""
PG-MambaGAN — Standard Losses (L1, Wasserstein, Gradient Penalty)

Core GAN loss functions for WGAN-GP training.

Usage:
    from models.losses.standard_loss import l1_loss, wasserstein_g_loss, \
        wasserstein_d_loss, gradient_penalty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# L1 Reconstruction Loss
# ======================================================================

class L1Loss(nn.Module):
    """Pixel-wise L1 (MAE) loss."""
    
    def forward(
        self, predicted: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return F.l1_loss(predicted, target)


# ======================================================================
# Wasserstein Loss
# ======================================================================

def wasserstein_g_loss(d_fake: torch.Tensor) -> torch.Tensor:
    """
    Generator's Wasserstein loss.
    G wants D(fake) to be as HIGH as possible.
    L_G = -E[D(G(x))]
    """
    return -d_fake.mean()


def wasserstein_d_loss(
    d_real: torch.Tensor, d_fake: torch.Tensor
) -> torch.Tensor:
    """
    Discriminator's Wasserstein loss.
    D wants D(real) HIGH and D(fake) LOW.
    L_D = E[D(fake)] - E[D(real)]
    """
    return d_fake.mean() - d_real.mean()


# ======================================================================
# Gradient Penalty (WGAN-GP)
# ======================================================================

def gradient_penalty(
    discriminator: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    condition: torch.Tensor,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    """
    Compute gradient penalty for WGAN-GP.
    
    Enforces the Lipschitz constraint by penalizing the gradient norm
    of the discriminator at interpolated points between real and fake.
    
    Args:
        discriminator: PatchGAN discriminator module.
        real: Real NDCT images (B, 1, H, W).
        fake: Fake/predicted images (B, 1, H, W).
        condition: Conditioning LDCT images (B, 1, H, W).
        lambda_gp: Gradient penalty coefficient (default: 10.0).
        
    Returns:
        Gradient penalty loss (scalar).
    """
    B = real.shape[0]
    device = real.device
    
    # Random interpolation factor
    alpha = torch.rand(B, 1, 1, 1, device=device)
    
    # Interpolated samples
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    
    # Discriminator on interpolated (concat with condition)
    d_input = torch.cat([condition, interpolated], dim=1)
    d_interpolated = discriminator(d_input)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # Gradient norm penalty
    gradients = gradients.view(B, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = lambda_gp * ((gradient_norm - 1.0) ** 2).mean()
    
    return penalty
